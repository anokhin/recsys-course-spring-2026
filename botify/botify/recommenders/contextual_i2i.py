import json
import pickle
import random

from .recommender import Recommender


class ContextualI2I(Recommender):
    """
    Improved SasRec I2I recommender with recency-biased anchor selection.

    Key difference from baseline: instead of sampling anchors weighted
    by total listen time, we strongly favor the most recent track(s).
    This captures short-term user intent more accurately.

    Method:
    1. Sort history by recency (most recent first)
    2. Assign exponentially decaying weights: w_i = 2^(-i) for position i
    3. Pick anchor with probability proportional to w_i * listen_time
    4. Use SasRec I2I recommendations for the chosen anchor
    """

    def __init__(self, listen_history_redis, sasrec_redis, fallback):
        self.listen_history_redis = listen_history_redis
        self.sasrec_redis = sasrec_redis
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)

        if not history:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        # Sort by recency (list is in order: newest first)
        # Apply exponential decay for recency weights
        weights = []
        for i, (track, listened_time) in enumerate(history):
            recency_weight = 2.0 ** (-i)  # exponential decay
            weights.append(max(0.01, listened_time * recency_weight))

        n_anchors = min(3, len(history))
        anchor_indices = sorted(
            range(len(history)),
            key=lambda i: -weights[i]
        )[:n_anchors]

        for idx in anchor_indices:
            anchor = history[idx][0]
            candidate = self._recommend_from_anchor(anchor, seen_tracks)
            if candidate is not None:
                return candidate

        return self.fallback.recommend_next(user, prev_track, prev_track_time)

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
        data = self.sasrec_redis.get(anchor)
        if data is None:
            return None
        recommendations = pickle.loads(data)
        for track in recommendations:
            candidate = int(track)
            if candidate not in seen_tracks:
                return candidate
        return None