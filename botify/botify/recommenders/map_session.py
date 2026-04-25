import json
import pickle
from collections import defaultdict

import numpy as np

from .recommender import Recommender


class MAPSession(Recommender):
    def __init__(
        self,
        listen_history_redis,
        sasrec_i2i_redis,
        embeddings_path: str,
        fallback,
        lam: float = 1.0,
        k_candidates: int = 20,
        min_time: float = 0.1,
    ):
        self.listen_history_redis = listen_history_redis
        self.sasrec_i2i_redis = sasrec_i2i_redis
        self.fallback = fallback
        self.lam = float(lam)
        self.k_candidates = int(k_candidates)
        self.min_time = float(min_time)
        self.E = self._load_embeddings(embeddings_path)

    @staticmethod
    def _load_embeddings(path: str) -> np.ndarray:
        rows = []
        max_id = -1
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                rows.append((int(rec["item_id"]), rec["embedding"]))
                if rec["item_id"] > max_id:
                    max_id = int(rec["item_id"])
        d = len(rows[0][1])
        E = np.zeros((max_id + 1, d), dtype=np.float32)
        for item_id, vec in rows:
            E[item_id] = vec
        return E

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        if not history:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        seen_tracks = set(track for track, _ in history)

        active = [(t, time) for t, time in history if time >= self.min_time]
        if len(active) < 2:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        first_track = history[-1][0]
        track_ids = np.asarray([t for t, _ in active], dtype=np.int64)
        times = np.asarray([time for _, time in active], dtype=np.float32)

        X = self.E[track_ids]
        theta_prior = self.E[first_track]

        theta_star = self._solve_map(X, times, theta_prior)
        if theta_star is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        anchor = self._best_anchor(history)
        candidates = self._candidates(anchor, seen_tracks)
        if not candidates:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        cand_arr = np.asarray(candidates, dtype=np.int64)
        scores = self.E[cand_arr] @ theta_star
        return int(cand_arr[int(np.argmax(scores))])

    def _load_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _best_anchor(self, history):
        track_time = defaultdict(float)
        for track, time in history:
            track_time[track] += time
        return max(track_time.items(), key=lambda kv: kv[1])[0]

    def _candidates(self, anchor, seen_tracks):
        data = self.sasrec_i2i_redis.get(anchor)
        if data is None:
            return []
        recommendations = pickle.loads(data)
        out = []
        for track in recommendations[: self.k_candidates]:
            candidate = int(track)
            if candidate not in seen_tracks:
                out.append(candidate)
        return out

    def _solve_map(self, X: np.ndarray, times: np.ndarray, theta_prior: np.ndarray):
        clipped = np.clip(times, 0.01, 0.99)
        y = np.log(clipped / (1.0 - clipped)).astype(np.float32)
        w = times.astype(np.float32)

        Xw = X * w[:, None]
        d = X.shape[1]
        A = Xw.T @ X + self.lam * np.eye(d, dtype=np.float32)
        b = Xw.T @ y + self.lam * theta_prior
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
