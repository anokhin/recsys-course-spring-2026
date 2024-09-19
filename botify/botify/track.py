import itertools
import json
import pickle
from dataclasses import dataclass, field
from typing import List


@dataclass
class Track:
    track: int
    artist: str
    title: str
    recommendations: List[int] = field(default=lambda: [])


class Catalog:
    """
    A helper class used to load track data upon server startup
    and store the data to redis.
    """

    def __init__(self, app):
        self.app = app
        self.tracks = []
        self.top_tracks = []

    def load(self, catalog_path):
        self.app.logger.info(f"Loading tracks from {catalog_path}")
        with open(catalog_path) as catalog_file:
            for j, line in enumerate(catalog_file):
                data = json.loads(line)
                self.tracks.append(
                    Track(
                        data["track"],
                        data["artist"],
                        data["title"],
                        data.get("recommendations", []),
                    )
                )
        self.app.logger.info(f"Loaded {j + 1} tracks")
        return self

    def upload_tracks(self, redis_tracks):
        self.app.logger.info(f"Uploading tracks to redis")
        for track in self.tracks:
            redis_tracks.set(track.track, self.to_bytes(track))

    def upload_artists(self, redis):
        self.app.logger.info(f"Uploading artists to redis")
        uploaded = 0
        sorted_tracks = sorted(self.tracks, key=lambda track: track.artist)
        for artist, tracks in itertools.groupby(sorted_tracks, key=lambda track: track.artist):
            redis.set(artist, self.to_bytes([track.track for track in tracks]))
            uploaded += 1

        self.app.logger.info(f"Uploaded {uploaded} artists")

    def to_bytes(self, instance):
        return pickle.dumps(instance)

    def from_bytes(self, bts):
        return pickle.loads(bts)
