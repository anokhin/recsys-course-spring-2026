import json
import pickle
import random

from .recommender import Recommender


class TopKRandomI2I(Recommender):
    """
    Selects randomly from top-K SasRec I2I candidates weighted by rank.
    Introduces controlled diversity vs deterministic top-1 selection.
    Uses SasRec ML model for candidate generation.
    """

    def __init__(self, listen_history_redis, i2i_redis, fallback_recommender, k=5):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback_recommender = fallback_recommender
        self.k = k

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)

        candidates = []
        weights = []

        data = self.i2i_redis.get(prev_track)
        if data is not None:
            recs = pickle.loads(data)
            for rank, track in enumerate(recs):
                candidate = int(track)
                if candidate not in seen_tracks:
                    candidates.append(candidate)
                    weights.append(1.0 / (rank + 1))
                if len(candidates) >= self.k:
                    break

        if candidates:
            return random.choices(candidates, weights=weights, k=1)[0]

        for track, _ in history:
            data = self.i2i_redis.get(track)
            if data is None:
                continue
            recs = pickle.loads(data)
            for track2 in recs:
                candidate = int(track2)
                if candidate not in seen_tracks:
                    return candidate

        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

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
