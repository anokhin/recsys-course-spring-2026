import random

from .recommender import Recommender


class Random(Recommender):
    def __init__(self, track_source):
        self.track_source = track_source

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        if hasattr(self.track_source, "randomkey"):
            return int(self.track_source.randomkey())
        return int(random.choice(self.track_source))
