import json
import pickle
import random
from .recommender import Recommender


class EnsembleRecommender(Recommender):
    def __init__(self, listen_history_redis, user_redis, item_redis, sasrec_redis, fallback):
        self.listen_history_redis = listen_history_redis
        self.item_redis = item_redis
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._history(user)
        if not history:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        tracks = [track for track, _ in history]
        weights = [max(time, 0.01) for _, time in history]
        anchor = random.choices(tracks, weights=weights, k=1)[0]
        seen = set(tracks)

        for candidate in self._load(self.item_redis, anchor):
            candidate = int(candidate)
            if candidate != prev_track and candidate not in seen:
                return candidate

        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    def _history(self, user):
        raw_entries = self.listen_history_redis.lrange(f"user:{user}:listens", 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _load(self, redis, key):
        raw = redis.get(key)
        if raw is None:
            return []
        return pickle.loads(raw)
