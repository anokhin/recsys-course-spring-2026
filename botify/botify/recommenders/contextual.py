from .recommender import Recommender
import json

class ContextualRecommender(Recommender):
    def __init__(self, recommendations_redis, history_redis, catalog, fallback):
        self.recommendations_redis = recommendations_redis
        self.history_redis = history_redis
        self.catalog = catalog
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        # 1. Достаем рекомендации, которые мы насчитали нашей ML моделью (HSTU)
        recommendations_data = self.recommendations_redis.get(user)
        if recommendations_data is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        recommendations = self.catalog.from_bytes(recommendations_data)

        # 2. Достаем историю прослушивания пользователя в этой сессии из Redis
        history_data = self.history_redis.get(f"user:{user}:listens")
        listened_tracks = set()
        if history_data:
            # Redis хранит историю как список JSON-строк
            history = self.history_redis.lrange(f"user:{user}:listens", 0, -1)
            for item in history:
                listened_tracks.add(json.loads(item)["track"])

        # 3. Фильтруем рекомендации: убираем то, что уже слушали, 
        # и выбираем лучший следующий трек
        for track in recommendations:
            if track not in listened_tracks:
                return track

        # 4. Если всё из рекомендаций уже послушали, отдаем фолбэк (рандом или i2i)
        return self.fallback.recommend_next(user, prev_track, prev_track_time)