from .recommender import Recommender
from botify.recommenders.random import Random
from botify.track import Catalog
import random

class StickyArtistMixed(Recommender):
    def __init__(self, track_redis, artist_redis, catalog: Catalog, random_recommender:Random):
        self.track_redis = track_redis
        self.artist_redis = artist_redis
        self.catalog = catalog
        self.random_recommender = random_recommender

    def recommend_next(self, user, prev_track, prev_track_time):
        if prev_track_time < 0.3:
            return self.random_recommender.recommend_next(user, prev_track, prev_track_time)

        track_bytes = self.track_redis.get(prev_track)
        if track_bytes is None:
            return int(self.track_redis.randomkey())

        track = self.catalog.from_bytes(track_bytes)
        artist = track.artist

        artist_bytes = self.artist_redis.get(artist)
        if artist_bytes is None:
            return int(self.track_redis.randomkey())

        artist_tracks = self.catalog.from_bytes(artist_bytes)  # список track_id
        if not artist_tracks:
            return int(self.track_redis.randomkey())

        candidates = [t for t in artist_tracks if t != prev_track] or artist_tracks
        return int(random.choice(candidates))



