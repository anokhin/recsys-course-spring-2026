import json
import pickle
import random
from collections import defaultdict

from .recommender import Recommender


class EnsembleI2I(Recommender):
    """
    Ensemble recommender that combines LightFM I2I and SasRec I2I models.

    For each anchor track from user history (weighted by listen time),
    it retrieves recommendations from both models and merges them
    via round-robin interleaving. This leverages complementary signals
    from collaborative filtering (LightFM) and sequential pattern learning
    (SasRec).

    If the candidate list is empty, falls back to a random recommender.
    """

    def __init__(self, listen_history_redis, lightfm_redis, sasrec_redis, fallback):
        self.listen_history_redis = listen_history_redis
        self.lightfm_redis = lightfm_redis
        self.sasrec_redis = sasrec_redis
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)

        if not history:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        # Build weighted anchor tracks
        track_time = defaultdict(float)
        for track, listened_time in history:
            track_time[track] += listened_time

        anchors = list(track_time.keys())
        weights = [track_time[t] for t in anchors]

        while anchors:
            # Pick anchor weighted by listen time
            anchor = random.choices(anchors, weights=weights, k=1)[0]

            # Get candidates from both models
            candidates = self._get_interleaved_candidates(anchor, seen_tracks)
            if candidates:
                candidate = candidates[0]
                if candidate not in seen_tracks:
                    return candidate

            # Remove anchor and retry
            anchor_idx = anchors.index(anchor)
            anchors.pop(anchor_idx)
            weights.pop(anchor_idx)

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

    def _get_i2i_candidates(self, redis_client, anchor: int, seen_tracks):
        """Get candidates from a single I2I model"""
        data = redis_client.get(anchor)
        if data is None:
            return []
        recommendations = pickle.loads(data)
        return [int(t) for t in recommendations if int(t) not in seen_tracks]

    def _get_interleaved_candidates(self, anchor: int, seen_tracks):
        """
        Merge candidates from both models via round-robin interleaving.
        This ensures diversity and coverage from both collaborative
        filtering (LightFM) and sequential (SasRec) signals.
        """
        lightfm_cands = self._get_i2i_candidates(self.lightfm_redis, anchor, seen_tracks)
        sasrec_cands = self._get_i2i_candidates(self.sasrec_redis, anchor, seen_tracks)

        merged = []
        max_len = max(len(lightfm_cands), len(sasrec_cands))
        for i in range(max_len):
            if i < len(lightfm_cands):
                merged.append(lightfm_cands[i])
            if i < len(sasrec_cands):
                merged.append(sasrec_cands[i])

        return merged