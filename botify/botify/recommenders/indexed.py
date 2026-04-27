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
            recs = list(self.catalog.from_bytes(recommendations))
            # Pick from top-10 to preserve ranking while allowing variety
            top_k = recs[:10]
            candidates = [t for t in top_k if t != prev_track] or top_k
            return int(random.choice(candidates))
        else:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
