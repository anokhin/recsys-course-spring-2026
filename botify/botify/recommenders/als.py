import json
import pickle
from .recommender import Recommender


class ALSRecommender(Recommender):
    def __init__(self, recommendations_redis, listen_history_redis, fallback):
        self.recommendations_redis = recommendations_redis
        self.listen_history_redis = listen_history_redis
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        data = self.recommendations_redis.get(user)
        if data is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        tracks = pickle.loads(data)
        if not tracks:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        seen = self._get_seen_tracks(user)

        for track in tracks:
            if int(track) not in seen:
                return int(track)
        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    def _get_seen_tracks(self, user: int) -> set:
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        seen = set()
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            seen.add(int(entry["track"]))
        return seen