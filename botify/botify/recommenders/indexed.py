import random

from .recommender import Recommender


class Indexed(Recommender):
    def __init__(self, recommendations_redis, catalog, fallback):
        self.recommendations_redis = recommendations_redis
        self.fallback = fallback
        self.catalog = catalog

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        recommendations = self.recommendations_redis.get(user)

        if recommendations is not None:
            shuffled = list(self.catalog.from_bytes(recommendations))
            if shuffled:
                return shuffled[0]

        rec = self.fallback.recommend_next(user, prev_track, prev_track_time)
        if rec is None or not isinstance(rec, int):
            return 0
        return rec
