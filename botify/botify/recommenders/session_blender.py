"""Online session-aware blender of SASRec and LightFM item-to-item tables.

For every request we read the user's listen history (up to ten most recent
entries maintained by the server) and score every i2i neighbour of every
history track. The score combines four signals:

    score(c) = Σ_anchor_a  W_m · pw_rank(c, a, m) · engage(a) · recency(a)
              × artist_discount(c)

where
  * ``W_m`` is a learned per-model weight produced by the offline
    leave-one-out OLS in ``jupyter/train_blender.py`` — the reranker reuses
    the ``sasrec`` and ``lightfm`` coefficients from ``blender_weights.json``.
  * ``pw_rank`` is the usual DCG-style logarithmic decay over neighbour rank.
  * ``engage(a) = 0.2 + listen_time(a)`` softly floors "skip" anchors while
    boosting ones the user actually enjoyed.
  * ``recency(a) = 0.85 ** (distance_from_latest)`` — newer anchors dominate.
  * ``artist_discount(c) = 0.7 ** artist_count_in_history(c)`` — each prior
    play of the same artist in the current session shrinks the candidate
    score, which discourages back-to-back repeats the simulator penalises.

Everything is computed per request from live state — no precomputed per-user
file, no batch model, no hand-crafted metadata features — so the treatment
adapts to whatever the user is doing inside the session.
"""

import json
import math
import pickle
from collections import defaultdict

from .recommender import Recommender


class SessionBlender(Recommender):
    RECENCY_DECAY = 0.85
    ENGAGE_FLOOR = 0.2
    ARTIST_DISCOUNT_BASE = 0.7
    # Upper bound on how much history we actually sweep for anchors — the
    # server already trims history to 10 entries, this is just a guard.
    MAX_ANCHORS = 10

    def __init__(
        self,
        listen_history_redis,
        tracks_redis,
        sasrec_i2i_redis,
        lightfm_i2i_redis,
        catalog,
        weights,
        fallback,
    ):
        self.listen_history_redis = listen_history_redis
        self.tracks_redis = tracks_redis
        self.sasrec_i2i_redis = sasrec_i2i_redis
        self.lightfm_i2i_redis = lightfm_i2i_redis
        self.catalog = catalog
        self.w_sasrec = float(weights.get("sasrec", 1.0))
        self.w_lightfm = float(weights.get("lightfm", 0.5))
        self.fallback = fallback

    def recommend_next(self, user, prev_track, prev_track_time):
        history = self._load_history(user)
        if not history:
            return self.fallback.recommend_next(
                user, prev_track, prev_track_time
            )

        seen = {track for track, _ in history}
        artist_counts = self._history_artist_counts(history)

        scores = defaultdict(float)
        n = len(history)
        newest_index = n - 1
        for idx, (anchor, listen_time) in enumerate(history):
            recency = self.RECENCY_DECAY ** (newest_index - idx)
            engagement = self.ENGAGE_FLOOR + max(listen_time, 0.0)
            anchor_weight = recency * engagement

            self._accumulate(
                scores, seen,
                self.sasrec_i2i_redis, anchor,
                self.w_sasrec * anchor_weight,
            )
            self._accumulate(
                scores, seen,
                self.lightfm_i2i_redis, anchor,
                self.w_lightfm * anchor_weight,
            )

        if not scores:
            return self.fallback.recommend_next(
                user, prev_track, prev_track_time
            )

        for cand in list(scores.keys()):
            artist = self._artist_of(cand)
            if artist is None:
                continue
            repeats = artist_counts.get(artist, 0)
            if repeats:
                scores[cand] *= self.ARTIST_DISCOUNT_BASE ** repeats

        best_track = max(scores.items(), key=lambda kv: kv[1])[0]
        return int(best_track)

    def _accumulate(self, scores, seen, i2i_redis, anchor, base_weight):
        neighbours = self._i2i_neighbours(i2i_redis, anchor)
        for rank, cand in enumerate(neighbours):
            if cand in seen:
                continue
            scores[cand] += base_weight / math.log2(2 + rank)

    def _i2i_neighbours(self, redis_conn, track):
        data = redis_conn.get(track)
        if data is None:
            return ()
        try:
            return [int(t) for t in pickle.loads(data)]
        except Exception:
            return ()

    def _load_history(self, user):
        raw = self.listen_history_redis.lrange(
            f"user:{user}:listens", 0, self.MAX_ANCHORS - 1
        )
        entries = []
        for r in raw:
            if isinstance(r, bytes):
                r = r.decode("utf-8")
            payload = json.loads(r)
            entries.append(
                (int(payload["track"]), float(payload["time"]))
            )
        # lpush stores newest-first; reverse so iteration runs oldest→newest.
        entries.reverse()
        return entries

    def _history_artist_counts(self, history):
        counts = defaultdict(int)
        for track, _ in history:
            artist = self._artist_of(track)
            if artist is not None:
                counts[artist] += 1
        return counts

    def _artist_of(self, track):
        raw = self.tracks_redis.get(track)
        if raw is None:
            return None
        try:
            record = self.catalog.from_bytes(raw)
        except Exception:
            return None
        return getattr(record, "artist", None)
