import json
import pickle
from collections import defaultdict

from .recommender import Recommender


class Item2VecAggRecommender(Recommender):
    """
    ML recommender based on Item2Vec (Word2Vec on track sequences).
    Trained on real user listening sessions from botify logs.
    Aggregates Item2Vec candidates across full session history
    weighted by listen time for better personalization.
    """

    def __init__(self, listen_history_redis, i2i_redis, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback_recommender = fallback_recommender

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)

        track_time = defaultdict(float)
        for track, listened_time in history:
            track_time[track] += listened_time

        scores = defaultdict(float)

        self._score_anchor(prev_track, 10.0, seen_tracks, scores)

        for anchor, time_weight in track_time.items():
            self._score_anchor(anchor, time_weight, seen_tracks, scores)

        if scores:
            return max(scores, key=scores.get)

        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

    def _score_anchor(self, anchor, weight, seen_tracks, scores):
        data = self.i2i_redis.get(anchor)
        if data is None:
            return
        recommendations = pickle.loads(data)
        for rank, track in enumerate(recommendations):
            candidate = int(track)
            if candidate not in seen_tracks:
                scores[candidate] += weight / (rank + 1)

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
