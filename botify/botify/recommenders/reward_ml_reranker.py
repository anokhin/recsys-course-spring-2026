import json
import math
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .recommender import Recommender

_MAX_I2I_RANK = 64


class RewardMlRerankerRecommender(Recommender):
    def __init__(
        self,
        listen_history_redis,
        candidate_i2i_redis,
        fallback_recommender,
        model_json_path: str,
        tracks_catalog_path: str,
    ):
        self.listen_history_redis = listen_history_redis
        self.candidate_i2i_redis = candidate_i2i_redis
        self.fallback_recommender = fallback_recommender

        with open(model_json_path, "r", encoding="utf-8") as f:
            model = json.load(f)

        self.coef = [float(x) for x in model["coef"]]
        self.intercept = float(model["intercept"])
        self.mean = [float(x) for x in model["mean"]]
        self.scale = [float(x) for x in model["scale"]]
        self.global_counts = model.get("global_counts")
        if self.global_counts is not None:
            self.global_counts = [float(x) for x in self.global_counts]

        self.artist_ids: Dict[int, int] = {}
        self.genres: Dict[int, str] = {}
        self._load_track_meta(tracks_catalog_path)

    def _load_track_meta(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                tid = int(row["track"])
                self.artist_ids[tid] = int(row.get("artist_id", -1))
                glist = row.get("genres") or []
                self.genres[tid] = str(glist[0]) if glist else "unknown"

    def _log_pop(self, track: int) -> float:
        if not self.global_counts or track < 0 or track >= len(self.global_counts):
            return 0.0
        c = self.global_counts[track]
        if c <= 0.0:
            return 0.0
        return math.log1p(c)

    def _feats(self, prev: int, cand: int, rank: int) -> List[float]:
        ap = self.artist_ids.get(prev, -1)
        ac = self.artist_ids.get(cand, -2)
        gp = self.genres.get(prev, "")
        gc = self.genres.get(cand, "")
        return [
            1.0 / (1.0 + rank),
            min(rank, 99) / 99.0,
            1.0 if rank < 5 else 0.0,
            self._log_pop(cand),
            1.0 if ap == ac and ap >= 0 else 0.0,
            1.0 if gp == gc and gp else 0.0,
        ]

    def _score(self, feats: List[float]) -> float:
        z = [(feats[i] - self.mean[i]) / self.scale[i] for i in range(len(feats))]
        s = self.intercept
        for i, w in enumerate(self.coef):
            s += w * z[i]
        return s

    @staticmethod
    def _pop_anchor(anchors: List[int], weights: List[float], anchor: int) -> None:
        idx = anchors.index(anchor)
        anchors.pop(idx)
        weights.pop(idx)

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen = {t for t, _ in history}

        track_time = defaultdict(float)
        for track, listened_time in history:
            track_time[track] += listened_time
        anchors = list(track_time.keys())
        weights = [max(track_time[t], 1e-6) for t in anchors]

        if not anchors:
            return self.fallback_recommender.recommend_next(
                user, prev_track, prev_track_time
            )

        anchors_work = anchors[:]
        weights_work = weights[:]
        while anchors_work:
            anchor = random.choices(anchors_work, weights=weights_work, k=1)[0]
            pool = self._load_i2i_candidates(anchor)
            if not pool:
                self._pop_anchor(anchors_work, weights_work, anchor)
                continue

            best_track: Optional[int] = None
            best_score = float("-inf")
            for rank, cand in enumerate(pool[:_MAX_I2I_RANK]):
                if cand in seen:
                    continue
                sc = self._score(self._feats(anchor, cand, rank))
                if sc > best_score:
                    best_score = sc
                    best_track = cand

            if best_track is not None:
                return best_track

            self._pop_anchor(anchors_work, weights_work, anchor)

        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

    def _load_user_history(self, user: int) -> List[Tuple[int, float]]:
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _load_i2i_candidates(self, key: int) -> List[int]:
        data = self.candidate_i2i_redis.get(key)
        if data is None:
            return []
        try:
            return [int(track) for track in pickle.loads(data)]
        except Exception:
            return []
