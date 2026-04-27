import json
import pickle

from .recommender import Recommender


class PrevTrackLightFM(Recommender):
    """
    Uses prev_track as primary anchor into LightFM I2I index.
    LightFM is a matrix factorization ML model - different from SasRec.
    """

    def __init__(self, listen_history_redis, lightfm_redis, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.lightfm_redis = lightfm_redis
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

    def _recommend_from_anchor(self, anchor: int, seen_tracks):
        data = self.lightfm_redis.get(anchor)
        if data is None:
            return None
        recommendations = pickle.loads(data)
        for track in recommendations:
            candidate = int(track)
            if candidate not in seen_tracks:
                return candidate
        return None
