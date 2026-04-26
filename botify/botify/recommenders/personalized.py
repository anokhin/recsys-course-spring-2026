import json
import pickle

from .recommender import Recommender


class PersonalizedRecommender(Recommender):
    def __init__(self, listen_history_redis, recommendations_redis, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.recommendations_redis = recommendations_redis
        self.fallback_recommender = fallback_recommender

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        seen_tracks = self._load_seen_tracks(user)
        for key in [
            "track:{0}".format(prev_track),
            "global",
        ]:
            payload = self._load_payload(key)
            if payload is None:
                continue
            for candidate in payload["tracks"]:
                candidate = int(candidate)
                if candidate not in seen_tracks:
                    return candidate

        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

    def _load_seen_tracks(self, user: int):
        raw_entries = self.listen_history_redis.lrange("user:{0}:listens".format(user), 0, -1)
        seen_tracks = set()
        for raw_entry in raw_entries:
            if isinstance(raw_entry, bytes):
                raw_entry = raw_entry.decode("utf-8")
            entry = json.loads(raw_entry)
            seen_tracks.add(int(entry["track"]))
        return seen_tracks

    def _load_payload(self, key):
        data = self.recommendations_redis.get(key)
        if data is None:
            return None
        return pickle.loads(data)
