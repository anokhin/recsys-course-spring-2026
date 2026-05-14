import random

from .recommender import Recommender


class Indexed(Recommender):
    def __init__(self, recommendations_redis, catalog, fallback, redis):
        self.recommendations_redis = recommendations_redis
        self.fallback = fallback
        self.catalog = catalog
        self.redis = redis  # 💡 нужен для shown

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        recommendations = self.recommendations_redis.get(user)

        if recommendations is not None:
            recs = list(self.catalog.from_bytes(recommendations))

            # === SHOWN ===
            shown_key = f"user:{user}:shown"
            shown_raw = self.redis.smembers(shown_key)
            shown = set(int(x) for x in shown_raw) if shown_raw else set()

            # === берем лучший НЕ показанный ===
            for track in recs[:50]:  # ограничиваем
                if track not in shown:
                    self.redis.sadd(shown_key, track)
                    return int(track)

        # fallback если всё уже показали
        return self.fallback.recommend_next(user, prev_track, prev_track_time)