import json
import random

from .recommender import Recommender


class Reranker(Recommender):
    """Serves a precomputed per-user reranked top-K list.

    The list in redis is already ordered by the offline blender (SASRec +
    EASE + HSTU + LightFM with position weights), so we preserve its order:
    filter out tracks the user has already heard, then sample from the top
    of the remaining list with a geometric bias toward the best rank.
    """

    TOP_BIAS = 0.6

    def __init__(self, listen_history_redis, recommendations_redis,
                 catalog, fallback):
        self.listen_history_redis = listen_history_redis
        self.recommendations_redis = recommendations_redis
        self.catalog = catalog
        self.fallback = fallback

    def recommend_next(self, user, prev_track, prev_track_time):
        recommendations = self.recommendations_redis.get(user)
        if recommendations is None:
            return self.fallback.recommend_next(
                user, prev_track, prev_track_time
            )

        ranked = list(self.catalog.from_bytes(recommendations))
        seen = self._load_seen(user)
        fresh = [int(t) for t in ranked if int(t) not in seen]

        if not fresh:
            return self.fallback.recommend_next(
                user, prev_track, prev_track_time
            )

        for track in fresh:
            if random.random() < self.TOP_BIAS:
                return track
        return fresh[-1]

    def _load_seen(self, user):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        seen = set()
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            seen.add(int(json.loads(raw)["track"]))
        return seen
