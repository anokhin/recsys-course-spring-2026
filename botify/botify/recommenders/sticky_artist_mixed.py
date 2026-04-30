from .recommender import Recommender
from botify.recommenders.random import Random
from botify.track import Catalog
import random


class StickyArtistMixed(Recommender):
    def __init__(self, track_redis, artist_redis, catalog: Catalog, random_recommender: Random):
        self.track_redis = track_redis
        self.artist_redis = artist_redis
        self.catalog = catalog
        self.random_recommender = random_recommender

    def recommend_next(self, user, prev_track, prev_track_time):
        if prev_track_time < 0.3:
            rec = self.random_recommender.recommend_next(
                user, prev_track, prev_track_time)
            if rec is not None and isinstance(rec, int):
                return rec
            return 0

        track_bytes = self.track_redis.get(prev_track)
        if track_bytes is None:
            key = self.track_redis.randomkey()
            if key is not None:
                return int(key)
            return 0

        track = self.catalog.from_bytes(track_bytes)
        artist = track.artist

        artist_bytes = self.artist_redis.get(artist)
        if artist_bytes is None:
            key = self.track_redis.randomkey()
            if key is not None:
                return int(key)
            return 0

        artist_tracks = self.catalog.from_bytes(artist_bytes)
        if not artist_tracks:
            key = self.track_redis.randomkey()
            if key is not None:
                return int(key)
            return 0

        candidates = [t for t in artist_tracks if t !=
                      prev_track] or artist_tracks
        return int(random.choice(candidates))
