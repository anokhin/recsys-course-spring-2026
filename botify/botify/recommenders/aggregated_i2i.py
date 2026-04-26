import json
import pickle
from collections import defaultdict

from .recommender import Recommender


class AggregatedI2I(Recommender):
    """
    Rank Fusion recommender.
    Aggregates SasRec I2I recommendations across ALL anchor tracks
    in user history using weighted scoring:
    score(candidate) = sum over anchors of listen_time / (rank + 1)
    Finds consensus best track across full listening history.
    """

    def __init__(self, listen_history_redis, i2i_redis, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback_recommender = fallback_recommender

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)

        if not history:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        track_time = defaultdict(float)
        for track, listened_time in history:
            track_time[track] += listened_time

        candidate_scores = defaultdict(float)

        for anchor, time_weight in track_time.items():
            data = self.i2i_redis.get(anchor)
            if data is None:
                continue
            recommendations = pickle.loads(data)
            for rank, track in enumerate(recommendations):
                candidate = int(track)
                if candidate not in seen_tracks:
                    candidate_scores[candidate] += time_weight / (rank + 1)

        if candidate_scores:
            return max(candidate_scores, key=candidate_scores.get)

        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

    def _load_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history
