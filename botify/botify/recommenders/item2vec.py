import json
import pickle

from .recommender import Recommender


class Item2VecRecommender(Recommender):
    """
    ML recommender based on Item2Vec (Word2Vec applied to track sequences).
    Trained on real user listening sessions from botify logs.
    Uses prev_track as anchor to find similar tracks via learned embeddings.
    """

    def __init__(self, listen_history_redis, i2i_redis, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback_recommender = fallback_recommender

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)

        candidate = self._recommend_from_anchor(prev_track, seen_tracks)
        if candidate is not None:
            return candidate

        for track, _ in history:
            candidate = self._recommend_from_anchor(track, seen_tracks)
            if candidate is not None:
                return candidate

        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

    def _recommend_from_anchor(self, anchor, seen_tracks):
        data = self.i2i_redis.get(anchor)
        if data is None:
            return None
        recommendations = pickle.loads(data)
        for track in recommendations:
            candidate = int(track)
            if candidate not in seen_tracks:
                return candidate
        return None

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
