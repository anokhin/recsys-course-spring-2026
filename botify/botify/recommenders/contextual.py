import json
import random
from .recommender import Recommender

class ContextualRecommender(Recommender):
    def __init__(self, tracks_redis, recs_hstu_redis, recs_sasrec_redis, history_redis, artists_redis, catalog, fallback):
        self.tracks_redis = tracks_redis
        self.recs_hstu_redis = recs_hstu_redis
        self.recs_sasrec_redis = recs_sasrec_redis
        self.history_redis = history_redis
        self.artists_redis = artists_redis
        self.catalog = catalog
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        try:
            # 1. История сессии — это наш главный буст
            history_key = f"user:{user}:listens"
            history_items = self.history_redis.lrange(history_key, 0, -1)
            listened = {json.loads(item)["track"] for item in history_items if item}

            # 2. ПРИОРИТЕТ 1: Персональные рекомендации HSTU
            # Если юзер есть в твоем файле — это даст лучший результат
            hstu_data = self.recs_hstu_redis.get(str(user))
            if hstu_data:
                hstu_recs = self.catalog.from_bytes(hstu_data)
                for t in hstu_recs:
                    if t not in listened:
                        return t

            # 3. ПРИОРИТЕТ 2: Sticky Artist (если трек дослушали > 0.5)
            # Это спасет, если юзера нет в HSTU
            if prev_track is not None and prev_track_time > 0.5:
                track_raw = self.tracks_redis.get(prev_track)
                if track_raw:
                    track_obj = self.catalog.from_bytes(track_raw)
                    artist_data = self.artists_redis.get(track_obj.artist)
                    if artist_data:
                        artist_tracks = list(self.catalog.from_bytes(artist_data))
                        random.shuffle(artist_tracks)
                        for t in artist_tracks:
                            if t not in listened:
                                return t

            # 4. ПРИОРИТЕТ 3: Фильтрованный SasRec
            # Если ничего выше не сработало, берем i2i базу, но БЕЗ ПОВТОРОВ
            if prev_track is not None:
                sas_data = self.recs_sasrec_redis.get(prev_track)
                if sas_data:
                    sas_recs = self.catalog.from_bytes(sas_data)
                    for t in sas_recs:
                        if t not in listened:
                            return t
                            
        except Exception as e:
            # Если где-то случился баг — падаем в надежный фолбэк
            pass
        
        return self.fallback.recommend_next(user, prev_track, prev_track_time)