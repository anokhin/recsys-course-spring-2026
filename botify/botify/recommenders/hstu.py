import json
import pickle
import random

from .recommender import Recommender


class HSTURecommender(Recommender):
    def __init__(self, listen_history_redis, hstu_redis, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.hstu_redis = hstu_redis
        self.fallback_recommender = fallback_recommender

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        data = self.hstu_redis.get(user)
        if data is None:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        tracks = pickle.loads(data)
        seen = self._load_seen_tracks(user)
        candidates = [int(t) for t in tracks if int(t) not in seen]

        if candidates:
            return random.choice(candidates)

        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

    def _load_seen_tracks(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        seen = set()
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            seen.add(int(entry["track"]))
        return seen

