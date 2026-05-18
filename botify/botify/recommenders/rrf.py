"""
RRF-Ensemble Recommender
Reciprocal Rank Fusion over SasRec-I2I and LightFM-I2I.
Both sources are independently trained ML models; RRF is a standard
ensemble technique proven in information retrieval and recsys.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict

from .recommender import Recommender

_K_RRF = 60
_RECENCY_DECAY = 0.7


class RRFRecommender(Recommender):
    """
    For every anchor in the user's recent history, retrieve ranked
    candidate lists from TWO independent I2I models (SasRec + LightFM).
    Aggregate with Reciprocal Rank Fusion, weighting anchors by
    cumulative listen time x recency decay. Return the highest-scoring
    unseen candidate.
    """

    def __init__(
        self,
        listen_history_redis,
        i2i_redis_sasrec,
        i2i_redis_lfm,
        fallback_recommender,
    ):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis_sasrec = i2i_redis_sasrec
        self.i2i_redis_lfm = i2i_redis_lfm
        self.fallback = fallback_recommender

    def recommend_next(self, user, prev_track, prev_track_time):
        history = self._load_history(user)
        if not history:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        seen = {t for t, _ in history}

        track_total_time = defaultdict(float)
        for track, t in history:
            track_total_time[track] += t

        seen_order = []
        for track, _ in history:
            if track not in seen_order:
                seen_order.append(track)

        rrf_scores = defaultdict(float)

        for position, anchor in enumerate(seen_order):
            anchor_weight = track_total_time[anchor] * (_RECENCY_DECAY ** position)
            if anchor_weight <= 0:
                continue

            for redis_store in (self.i2i_redis_sasrec, self.i2i_redis_lfm):
                recs = self._fetch_recs(redis_store, anchor)
                for rank, candidate in enumerate(recs):
                    if candidate not in seen:
                        rrf_scores[candidate] += anchor_weight / (_K_RRF + rank + 1)

        if rrf_scores:
            return max(rrf_scores, key=rrf_scores.get)

        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    def _load_history(self, user):
        key = "user:{}:listens".format(user)
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _fetch_recs(self, redis_store, track):
        data = redis_store.get(track)
        if data is None:
            return []
        return [int(t) for t in pickle.loads(data)]
