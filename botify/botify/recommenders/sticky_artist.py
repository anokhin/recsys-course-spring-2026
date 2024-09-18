import random

from .random import Random
from .recommender import Recommender


class StickyArtist(Recommender):
    def __init__(self, tracks_redis, artists_redis, catalog):
        self.fallback = Random(tracks_redis)

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        # TODO Seminar 1 step 4: implement the recommender
        return self.fallback.recommend_next(user, prev_track, prev_track_time)
