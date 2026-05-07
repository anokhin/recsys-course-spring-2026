import json

from .recommender import Recommender


class Indexed(Recommender):
    def __init__(self, listen_history_redis, recommendations_redis, catalog, fallback):
        self.listen_history_redis = listen_history_redis
        self.recommendations_redis = recommendations_redis
        self.fallback = fallback
        self.catalog = catalog

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        recommendations = self.recommendations_redis.get(user)

        if recommendations is not None:
            seen_tracks = self._load_seen_tracks(user)
            for track in self.catalog.from_bytes(recommendations):
                candidate = int(track)
                if candidate not in seen_tracks:
                    return candidate

        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    def _load_seen_tracks(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)

        seen_tracks = set()
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            seen_tracks.add(int(entry["track"]))
        return seen_tracks
