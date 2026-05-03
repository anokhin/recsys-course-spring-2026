import json
import random
import logging
from .recommender import Recommender

class Contextual(Recommender):
    def __init__(self, hstu_redis, catalog, i2i_recommender, fallback):
        self.hstu_redis = hstu_redis
        self.catalog = catalog
        self.i2i_recommender = i2i_recommender
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        if prev_track_time >= 0.55:
            return self.i2i_recommender.recommend_next(user, prev_track, prev_track_time)

        try:
            recommendations = self.hstu_redis.get(user)
            if recommendations:
                # Декодируем байты и парсим JSON
                hstu_tracks = json.loads(recommendations)
                
                if isinstance(hstu_tracks, list) and hstu_tracks:
                    top_tracks = hstu_tracks[:15]
                    
                    # Убираем текущий трек
                    if prev_track in top_tracks:
                        top_tracks.remove(prev_track)
                    
                    if top_tracks:
                        return random.choice(top_tracks)
        except Exception as e:
            pass
        
        return self.fallback.recommend_next(user, prev_track, prev_track_time)