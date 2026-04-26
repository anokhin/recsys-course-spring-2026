import json
import pickle
import random
from collections import defaultdict

from .recommender import Recommender


class I2IRecommender(Recommender):
    def __init__(self, listen_history_redis, i2i_redis, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback_recommender = fallback_recommender

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)

        if history:
            track_time = defaultdict(float)
            for track, listened_time in history:
                track_time[track] += listened_time

            anchors = list(track_time.keys())
            weights = [track_time[track] for track in anchors]

            while anchors:
                anchor = random.choices(anchors, weights=weights, k=1)[0]
                candidate = self._recommend_from_anchor(anchor, seen_tracks)
                if candidate is not None:
                    return candidate

                anchor_idx = anchors.index(anchor)
                anchors.pop(anchor_idx)
                weights.pop(anchor_idx)

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
        data = self.i2i_redis.get(anchor)
        if data is None:
            return None

        recommendations = pickle.loads(data)
        for track in recommendations:
            candidate = int(track)
            if candidate not in seen_tracks:
                return candidate
        return None
