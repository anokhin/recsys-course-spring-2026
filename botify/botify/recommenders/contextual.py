import json
import random
import math
from .recommender import Recommender

class Contextual(Recommender):
    def __init__(self, hstu_redis, catalog, i2i_recommender, fallback):
        self.hstu_redis = hstu_redis
        self.catalog = catalog
        self.i2i_recommender = i2i_recommender
        self.fallback = fallback

        self.w = -5.0
        self.b = 2.5
        
        self.user_history = {}

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        if user not in self.user_history:
            self.user_history[user] = []
        
        self.user_history[user].append(prev_track)
        
        if len(self.user_history[user]) > 20:
            self.user_history[user].pop(0)

        score = self.w * prev_track_time + self.b
        p_i2i = self.sigmoid(score)
        use_i2i = random.random() < p_i2i

        final_track = None

        if not use_i2i:
            recommendations = self.hstu_redis.get(user)
            if recommendations is not None:
                try:
                    if isinstance(recommendations, bytes):
                        recommendations = recommendations.decode("utf-8")

                    hstu_tracks = json.loads(recommendations)

                    if hstu_tracks:
                        filtered = [int(t) for t in hstu_tracks if int(t) not in self.user_history[user]]

                        if filtered:
                            final_track = filtered[0] 
                except Exception:
                    pass

        if final_track is None:
            final_track = self.i2i_recommender.recommend_next(user, prev_track, prev_track_time)
            
        self.user_history[user].append(final_track)

        return final_track