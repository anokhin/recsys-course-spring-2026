from .recommender import Recommender


class Random(Recommender):
    """Returns a random track id from the catalog.

    Falls back to ``prev_track`` if ``randomkey`` cannot produce a key, which
    can happen under transient Redis conditions and would otherwise crash the
    request with ``TypeError: int() argument must be ... not 'NoneType'``.
    """

    def __init__(self, track_redis):
        self.track_redis = track_redis

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        for _ in range(3):
            key = self.track_redis.randomkey()
            if key is None:
                continue
            try:
                return int(key)
            except (TypeError, ValueError):
                continue
        return int(prev_track)
