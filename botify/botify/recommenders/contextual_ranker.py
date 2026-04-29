from .recommender import Recommender

class ContextualRanker(Recommender):
    def __init__(self, recommendations_redis, track_data, fallback, catalog):
        self.recommendations_redis = recommendations_redis
        self.track_data = track_data
        self.fallback = fallback
        self.catalog = catalog

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        recs = self.recommendations_redis.get(user)
        if recs is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        candidates = list(self.catalog.from_bytes(recs))
        
        if prev_track is None or prev_track not in self.track_data or not candidates:
            return candidates[0] if candidates else self.fallback.recommend_next(user, prev_track, prev_track_time)

        prev_info = self.track_data[prev_track]
        prev_artist = prev_info.get("artist")

        if prev_track_time > 0.8:
            for t_id in candidates:
                t_info = self.track_data.get(t_id)
                if t_info and t_info.get("artist") == prev_artist and t_id != prev_track:
                    return t_id

        if prev_track_time < 0.2:
            index = min(10, len(candidates) - 1)
            return candidates[index]

        return candidates[0]