import json
from .recommender import Recommender

class ContextualRecommender(Recommender):
    def __init__(self, tracks_redis, recs_redis, history_redis, artists_redis, catalog, fallback):
        self.tracks_redis = tracks_redis
        self.recs_redis = recs_redis
        self.history_redis = history_redis
        self.artists_redis = artists_redis
        self.catalog = catalog
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        try:
            # 1. Получаем историю сессии (чтобы не повторяться)
            history_key = f"user:{user}:listens"
            history_items = self.history_redis.lrange(history_key, 0, -1)
            listened = set()
            for item in history_items:
                try:
                    listened.add(json.loads(item)["track"])
                except:
                    continue

            # 2. Sticky Artist: если прошлый трек дослушали (> 80%), суем того же артиста
            if prev_track_time > 0.8 and prev_track is not None:
                track_data_raw = self.tracks_redis.get(prev_track)
                if track_data_raw:
                    track_obj = self.catalog.from_bytes(track_data_raw)
                    artist_tracks_raw = self.artists_redis.get(track_obj.artist)
                    if artist_tracks_raw:
                        artist_tracks = self.catalog.from_bytes(artist_tracks_raw)
                        for t in artist_tracks:
                            if t not in listened:
                                return t

            # 3. SasRec: если Sticky не сработал, берем рекомендации SasRec
            if prev_track is not None:
                recs_raw = self.recs_redis.get(prev_track)
                if recs_raw:
                    recs = self.catalog.from_bytes(recs_raw)
                    for t in recs:
                        if t not in listened:
                            return t
        except Exception as e:
            pass
        
        # 4. Фолбэк на стандартный SasRec-I2I
        return self.fallback.recommend_next(user, prev_track, prev_track_time)