from .recommender import Recommender


class Random(Recommender):
    def __init__(self, track_redis):
        self.track_redis = track_redis

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        try:
            key = self.track_redis.randomkey()
            if key is not None:
                return int(key)
        except (TypeError, ValueError):
            pass
        return 0
