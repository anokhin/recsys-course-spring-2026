"""
Thompson Sampling Bandit Recommender.

Uses SasRec-I2I to generate a candidate set, then re-ranks candidates
using a per-track Beta(alpha, beta) distribution updated from observed
listening completion rates.  The model learns online during the simulation:
tracks users tend to complete (time >= threshold) accumulate alpha,
tracks users skip accumulate beta.  Thompson Sampling naturally balances
exploration and exploitation.

This is fundamentally different from static SasRec-I2I because it adapts
to the live simulator feedback rather than relying on pre-trained embeddings.
"""
from __future__ import annotations

import json
import pickle
import random
from collections import defaultdict

from .recommender import Recommender

_COMPLETION_THRESHOLD = 0.5   # listen fraction considered a "success"
_TOP_ANCHORS = 5              # how many history anchors to query


class TSBanditRecommender(Recommender):

    def __init__(self, listen_history_redis, i2i_redis, fallback_recommender):
        self.history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback = fallback_recommender

    # ------------------------------------------------------------------ #
    # Called for EVERY track listen (both groups) so the bandit warms up
    # as fast as possible.
    def update(self, track, listen_time):
        if listen_time >= _COMPLETION_THRESHOLD:
            self.history_redis.incr("bandit:a:{}".format(track))
        else:
            self.history_redis.incr("bandit:b:{}".format(track))

    # ------------------------------------------------------------------ #
    def recommend_next(self, user, prev_track, prev_track_time):
        history = self._load_history(user)
        if not history:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        seen = {t for t, _ in history}

        # Pick top-N anchors by total cumulative listen time
        track_time = defaultdict(float)
        for t, ti in history:
            track_time[t] += ti
        anchors = sorted(track_time, key=lambda x: -track_time[x])[:_TOP_ANCHORS]

        # Collect unseen candidates from SasRec I2I lists
        candidates = []
        seen_candidates = set()
        for anchor in anchors:
            for rec in self._get_i2i_recs(anchor):
                if rec not in seen and rec not in seen_candidates:
                    candidates.append(rec)
                    seen_candidates.add(rec)

        if not candidates:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        # Batch-fetch alpha/beta from Redis
        alpha_keys = ["bandit:a:{}".format(c) for c in candidates]
        beta_keys  = ["bandit:b:{}".format(c) for c in candidates]
        raw = self.history_redis.mget(alpha_keys + beta_keys)
        n = len(candidates)

        # Thompson Sampling: draw from Beta(alpha, beta) for each candidate
        best, best_val = None, -1.0
        for i, c in enumerate(candidates):
            a = float(raw[i]     or 1)
            b = float(raw[i + n] or 1)
            sample = random.betavariate(a, b)
            if sample > best_val:
                best_val = sample
                best = c

        return best

    # ------------------------------------------------------------------ #
    def _load_history(self, user):
        raw_entries = self.history_redis.lrange("user:{}:listens".format(user), 0, -1)
        result = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            result.append((int(entry["track"]), float(entry["time"])))
        return result

    def _get_i2i_recs(self, track):
        data = self.i2i_redis.get(track)
        if data is None:
            return []
        return [int(t) for t in pickle.loads(data)]
