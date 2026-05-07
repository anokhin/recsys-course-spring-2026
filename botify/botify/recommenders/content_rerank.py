from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from .recommender import Recommender


class ContentRerankRecommender(Recommender):

    DIVERSITY_PENALTY = 0.07
    MIN_LISTEN_TIME = 0.10
    CANDIDATES_PER_ANCHOR = 10
    MAX_HISTORY_LEN = 50
    MAX_ANCHORS = 8
    SKIP_PENALTY_THRESHOLD = 0.05

    def __init__(
        self,
        listen_history_redis,
        i2i_redis,
        embeddings_path: str,
        meta_path: str,
        fallback_recommender,
    ):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback_recommender = fallback_recommender

        self.embeddings = np.load(embeddings_path).astype(np.float32)
        with open(meta_path) as fh:
            meta = json.load(fh)
        self.dim = int(meta["dim"])
        artist_by_track = meta.get("artist_by_track", {})
        n = self.embeddings.shape[0]
        self.artist_by_track: list[str] = [""] * n
        for tid_str, artist in artist_by_track.items():
            tid = int(tid_str)
            if 0 <= tid < n:
                self.artist_by_track[tid] = artist or ""

        self._i2i_cache: dict[int, list[int] | None] = {}

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        if not history:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        seen_tracks = {track for track, _ in history}

        session_vec = self._session_vector(history)
        if session_vec is None:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        candidates = self._collect_candidates(history, seen_tracks)
        if not candidates:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        artist_counts: Counter[str] = Counter()
        for track, time in history:
            if 0 <= track < len(self.artist_by_track):
                artist_counts[self.artist_by_track[track]] += 1

        candidate_ids = np.fromiter(candidates, dtype=np.int64, count=len(candidates))
        embs = self.embeddings[candidate_ids]
        cos_scores = embs @ session_vec
        artist_pen = np.fromiter(
            (
                artist_counts.get(self.artist_by_track[c] if 0 <= c < len(self.artist_by_track) else "", 0)
                for c in candidate_ids
            ),
            dtype=np.float32,
            count=len(candidate_ids),
        )
        scores = cos_scores - self.DIVERSITY_PENALTY * artist_pen

        best_idx = int(np.argmax(scores))
        best_track = int(candidate_ids[best_idx])
        if best_track in seen_tracks:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)
        return best_track

    def _load_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, self.MAX_HISTORY_LEN - 1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _session_vector(self, history):
        n = self.embeddings.shape[0]
        acc = np.zeros(self.dim, dtype=np.float32)
        total_w = 0.0
        for track, time in history:
            if not (0 <= track < n):
                continue
            if time >= self.MIN_LISTEN_TIME:
                w = float(time)
                acc += self.embeddings[track] * w
                total_w += w
            elif time < self.SKIP_PENALTY_THRESHOLD:
                acc -= self.embeddings[track] * 0.2
        if total_w <= 0 and not np.any(acc):
            return None
        norm = float(np.linalg.norm(acc))
        if norm <= 1e-8:
            return None
        return (acc / norm).astype(np.float32)

    def _collect_candidates(self, history, seen_tracks):
        candidates: set[int] = set()
        ordered = sorted(history, key=lambda x: x[1], reverse=True)[: self.MAX_ANCHORS]
        for track, _ in ordered:
            recs = self._get_i2i(track)
            if not recs:
                continue
            for rid in recs[: self.CANDIDATES_PER_ANCHOR]:
                if rid not in seen_tracks:
                    candidates.add(rid)
        return candidates

    def _get_i2i(self, track: int):
        cached = self._i2i_cache.get(track)
        if cached is not None:
            return cached
        data = self.i2i_redis.get(track)
        if data is None:
            self._i2i_cache[track] = []
            return []
        try:
            recs = pickle.loads(data)
        except Exception:
            self._i2i_cache[track] = []
            return []
        recs_int = [int(r) for r in recs[: self.CANDIDATES_PER_ANCHOR]]
        self._i2i_cache[track] = recs_int
        return recs_int
