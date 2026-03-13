from collections import Counter
from dataclasses import dataclass
import numpy as np


@dataclass
class Playback:
    track: int
    time: float
    artist: str = None


class Session:
    def __init__(
        self, user: int, embedding: np.array, first_playback: Playback, budget: int
    ):
        self.user = user
        self.embedding = embedding
        self.budget = budget
        self.playback = [first_playback]
        self.seen_tracks = {first_playback.track}
        self.artist_counter = Counter([first_playback.artist])
        self.finished = False

    def observe(self):
        return {"user": self.user, "track": self.playback[-1].track}

    def update(self, playback: Playback, budget_decrement: int):
        self.playback.append(playback)
        self.seen_tracks.add(playback.track)
        self.artist_counter[playback.artist] += 1
        self.budget -= budget_decrement

    def finish(self):
        self.finished = True

    def artist_counts(self):
        return self.artist_counter

    def __contains__(self, track):
        return track in self.seen_tracks

    def __repr__(self):
        return (
            f"{self.user}:{self.playback}:{self.budget}" + "." if self.finished else ""
        )
