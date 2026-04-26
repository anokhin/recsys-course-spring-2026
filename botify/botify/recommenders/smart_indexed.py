import json
import random

from .recommender import Recommender


class SmartIndexed(Recommender):
    """
    Uses pre-computed ML recommendations (HSTU neural model) for each user,
    filtering out already-seen tracks from listen history.
    """

    def __init__(self, recommendations_redis, listen_history_redis, catalog, fallback):
        self.recommendations_redis = recommendations_redis
        self.listen_history_redis = listen_history_redis
        self.catalog = catalog
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)

        recommendations = self.recommendations_redis.get(user)
        if recommendations is not None:
            candidates = list(self.catalog.from_bytes(recommendations))
            unseen = [t for t in candidates if t not in seen_tracks]
            if unseen:
                return unseen[0]
            if candidates:
                return random.choice(candidates)

        return self.fallback.recommend_next(user, prev_track, prev_track_time)

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
