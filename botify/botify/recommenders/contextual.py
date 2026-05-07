import json
from .recommender import Recommender

class ContextualRecommender(Recommender):
    def __init__(self, recs_hstu_redis, recs_sasrec_redis, history_redis, catalog, fallback):
        self.recs_hstu_redis = recs_hstu_redis
        self.recs_sasrec_redis = recs_sasrec_redis
        self.history_redis = history_redis
        self.catalog = catalog
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        try:
            # 1. История сессии (фильтруем всё, что уже слушали)
            history_key = f"user:{user}:listens"
            history_items = self.history_redis.lrange(history_key, 0, -1)
            listened = set()
            if history_items:
                for item in history_items:
                    try:
                        listened.add(json.loads(item)["track"])
                    except: continue

            # 2. ПРИОРИТЕТ 1: Твои HSTU рекомендации (ML)
            # Пытаемся достать данные по юзеру
            hstu_data = self.recs_hstu_redis.get(user)
            if hstu_data is None:
                hstu_data = self.recs_hstu_redis.get(str(user))
            
            if hstu_data is not None:
                recommendations = self.catalog.from_bytes(hstu_data)
                for track in recommendations:
                    if track not in listened:
                        return track

            # 3. ПРИОРИТЕТ 2: Контекстный SasRec (из базы SasRec)
            # Если юзера нет в HSTU, используем базу SasRec, но с нашей фильтрацией истории
            if prev_track is not None:
                sas_data = self.recs_sasrec_redis.get(prev_track)
                if sas_data is None:
                    sas_data = self.recs_sasrec_redis.get(str(prev_track))
                
                if sas_data is not None:
                    recs = self.catalog.from_bytes(sas_data)
                    for track in recs:
                        if track not in listened:
                            return track
        except:
            # Если что-то пошло не так, вообще не паримся и идем в фолбэк
            pass

        # 4. НАДЕЖНЫЙ ФОЛБЭК: возвращаем то, что вернул бы базовый SasRec
        return self.fallback.recommend_next(user, prev_track, prev_track_time)