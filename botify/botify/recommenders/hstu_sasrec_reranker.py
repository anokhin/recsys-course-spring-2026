import json
import pickle
from collections import defaultdict

from .recommender import Recommender


class HSTUwithSasRecReranker(Recommender):
    """
    Two-stage ML recommender:
    Stage 1 (Retrieval): HSTU neural model generates personalized
                         candidate tracks per user.
    Stage 2 (Reranking): SasRec-I2I scores each candidate based on
                         session history anchors.
    Returns top-scored unseen candidate.
    """

    def __init__(self, hstu_redis, sasrec_redis, listen_history_redis, catalog, fallback):
        self.hstu_redis = hstu_redis
        self.sasrec_redis = sasrec_redis
        self.listen_history_redis = listen_history_redis
        self.catalog = catalog
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)

        # Stage 1: HSTU retrieval - get personalized candidates
        hstu_data = self.hstu_redis.get(user)
        if hstu_data is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        candidates = list(self.catalog.from_bytes(hstu_data))
        unseen_candidates = set(c for c in candidates if c not in seen_tracks)

        if not unseen_candidates:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        # Stage 2: SasRec-I2I reranking of HSTU candidates
        scores = defaultdict(float)

        # Score using prev_track as primary anchor
        self._score_from_anchor(prev_track, unseen_candidates, scores, weight=10.0)

        # Score using full session history
        track_time = defaultdict(float)
        for track, listened_time in history:
            track_time[track] += listened_time

        for anchor, time_weight in track_time.items():
            self._score_from_anchor(anchor, unseen_candidates, scores, weight=time_weight)

        if scores:
            return max(scores, key=scores.get)

        # Fallback: return first unseen HSTU candidate
        return next(iter(unseen_candidates))

    def _score_from_anchor(self, anchor, candidates, scores, weight):
        data = self.sasrec_redis.get(anchor)
        if data is None:
            return
        recommendations = pickle.loads(data)
        for rank, track in enumerate(recommendations):
            candidate = int(track)
            if candidate in candidates:
                scores[candidate] += weight / (rank + 1)

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
