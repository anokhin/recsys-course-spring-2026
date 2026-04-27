import json
import pickle

from .recommender import Recommender


class PrevTrackI2I(Recommender):
    """
    Uses the immediately previous track as the anchor for I2I lookup.
    This captures the most recent user intent rather than averaging history.
    Falls back to history-weighted I2I if no recommendations found.
    """

    def __init__(self, listen_history_redis, i2i_redis, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback_recommender = fallback_recommender

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)

        # Score candidates from prev_track + top history anchor
        from collections import defaultdict
        scores = defaultdict(float)

        recs = self._get_recommendations(prev_track)
        for rank, track in enumerate(recs):
            if track not in seen_tracks:
                scores[track] += 100.0 / (rank + 1)

        if history:
            best_anchor = max(history, key=lambda x: x[1])[0]
            recs2 = self._get_recommendations(best_anchor)
            for rank, track in enumerate(recs2):
                if track not in seen_tracks:
                    scores[track] += prev_track_time / (rank + 1)

        if scores:
            return max(scores, key=scores.get)

        # Fall back to history anchors by recency
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

    def _get_recommendations(self, anchor: int):
        data = self.i2i_redis.get(anchor)
        if data is None:
            return []
        return [int(t) for t in pickle.loads(data)]

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
