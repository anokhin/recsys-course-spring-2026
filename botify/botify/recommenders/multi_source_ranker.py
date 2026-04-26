import json
import pickle
from collections import defaultdict

from .recommender import Recommender

K_RRF = 60        # standard RRF constant
MAX_ANCHORS = 5   # use top-5 most-listened anchors from session history
LFM_WEIGHT = 0.7  # LightFM relative weight vs SasRec
HSTU_WEIGHT = 0.5 # HSTU user-based relative weight (long-term preference signal)
ARTIST_PENALTY = 0.25  # score multiplier when candidate's artist was recently heard
ARTIST_WINDOW = 3      # number of recent unique artists to track


class MultiSourceRanker(Recommender):
    """
    Ensemble ranker over SasRec-I2I, LightFM-I2I, and HSTU user recommendations.

    Improvement over SasRec-I2I baseline:
    1. Uses ALL recent anchors (not one random anchor) — lower variance.
    2. Fuses three complementary sources via Reciprocal Rank Fusion:
       SasRec-I2I (session context), LightFM-I2I (collaborative filtering),
       HSTU (long-term user preferences).
    3. Penalises candidates whose artist was recently heard in the session,
       directly countering the simulator's artist_discount_gamma penalty.
    Artist lookup uses an in-memory dict — zero extra Redis calls.
    """

    def __init__(
        self,
        listen_history_redis,
        sasrec_redis,
        lfm_redis,
        hstu_redis,
        track_artist_map: dict,
        fallback,
    ):
        self.listen_history_redis = listen_history_redis
        self.sasrec_redis = sasrec_redis
        self.lfm_redis = lfm_redis
        self.hstu_redis = hstu_redis
        self.track_artist_map = track_artist_map
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_history(user)
        if not history:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        seen = {track for track, _ in history}

        # Weight anchors by cumulative listen time
        track_time = defaultdict(float)
        for track, t in history:
            track_time[track] += t
        top_anchors = sorted(track_time.items(), key=lambda x: -x[1])[:MAX_ANCHORS]

        # RRF fusion: SasRec-I2I + LightFM-I2I (anchor-based)
        scores = defaultdict(float)
        for anchor, weight in top_anchors:
            for rank, track in enumerate(self._get_i2i(anchor, self.sasrec_redis)):
                if track not in seen:
                    scores[track] += weight / (rank + K_RRF)
            for rank, track in enumerate(self._get_i2i(anchor, self.lfm_redis)):
                if track not in seen:
                    scores[track] += LFM_WEIGHT * weight / (rank + K_RRF)

        # HSTU user-based recommendations (long-term preferences, user-level lookup)
        for rank, track in enumerate(self._get_user_recs(user)):
            if track not in seen:
                scores[track] += HSTU_WEIGHT / (rank + K_RRF)

        if not scores:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        recent_artists = self._recent_artists(history)

        def effective_score(track):
            artist = self.track_artist_map.get(track, "")
            penalty = ARTIST_PENALTY if artist in recent_artists else 1.0
            return scores[track] * penalty

        return max(scores, key=effective_score)

    # ------------------------------------------------------------------ #

    def _get_user_recs(self, user: int) -> list:
        data = self.hstu_redis.get(user)
        if data is None:
            return []
        return pickle.loads(data)

    def _get_i2i(self, anchor: int, redis_conn) -> list:
        data = redis_conn.get(anchor)
        if data is None:
            return []
        return pickle.loads(data)

    def _recent_artists(self, history) -> set:
        unique = []
        seen_artists = set()
        for track, _ in history:
            artist = self.track_artist_map.get(track, "")
            if artist and artist not in seen_artists:
                seen_artists.add(artist)
                unique.append(artist)
                if len(unique) >= ARTIST_WINDOW:
                    break
        return set(unique)

    def _load_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history
