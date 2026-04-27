from .recommender import Recommender
import random

class ContextualRanker(Recommender):
    def __init__(self, recommendations_redis, track_data, fallback):
        self.recommendations_redis = recommendations_redis
        self.track_data = track_data
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        recs = self.recommendations_redis.get(user)
        if recs is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        candidates = list(self.catalog.from_bytes(recs))
        
        if prev_track is None or prev_track not in self.track_data:
            return candidates[0]

        prev_info = self.track_data[prev_track]
        prev_mood = prev_info.get("mood")
        prev_genres = set(prev_info.get("genres", []))

        def get_score(t_id):
            t_info = self.track_data.get(t_id)
            if not t_info: return 0
            score = 0
            # Если предыдущий трек дослушали (time > 0.8), ищем такое же настроение
            if prev_track_time > 0.8:
                if t_info.get("mood") == prev_mood: score += 5
                score += len(prev_genres.intersection(set(t_info.get("genres", []))))
            elif prev_track_time < 0.2:
                if t_info.get("mood") == prev_mood: score -= 5
            return score

        candidates = sorted(candidates, key=get_score, reverse=True)
        return candidates[0]