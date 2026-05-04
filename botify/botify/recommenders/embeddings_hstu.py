import pickle
import joblib
import json
import random
from collections import defaultdict

from .recommender import Recommender


class EmbeddingHSTURecommender(Recommender):
    def __init__(self, track_redis, catalog, hstu_redis, listen_history_redis, fallback_recommender: Recommender, embeddings_path: str):
        self.track_redis = track_redis
        self.catalog = catalog
        self.hstu_redis = hstu_redis
        self.listen_history_redis = listen_history_redis
        self.fallback_recommender = fallback_recommender

        self.track_embeddings = joblib.load(embeddings_path)

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        if not history:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)
        seen_tracks = {t for t, _ in history}

        history_artist_counts = defaultdict(int)
        for track_id, _ in history:
            track_bytes = self.track_redis.get(track_id)
            if track_bytes is None:
                continue
            track = self.catalog.from_bytes(track_bytes)
            artist_id = track.artist_id
            if artist_id is not None:
                history_artist_counts[artist_id] += 1

        anchor_id = self._select_anchor(history, prev_track)
        anchor_vec = self.track_embeddings.get(anchor_id)
        if anchor_vec is None:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        raw = self.hstu_redis.get(f"hstu:{user}")
        if not raw:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        all_candidates = [int(t) for t in pickle.loads(raw)]
        candidates = [t for t in all_candidates if t not in seen_tracks]
        if not candidates:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        num_cands = len(candidates)

        SEMANTIC_WEIGHT = 0.65
        HSTU_RANK_WEIGHT = 0.35
        ARTIST_PENALTY_COEFF = 0.05

        best_track = None
        max_score = -float('inf')

        for i, cand_id in enumerate(candidates):
            cand_vec = self.track_embeddings.get(cand_id)
            if cand_vec is None:
                continue

            semantic_score = sum(float(a) * float(c) for a, c in zip(anchor_vec, cand_vec))
            hstu_rank_score = 1.0 - (i / (num_cands - 1)) if num_cands > 1 else 1.0

            cand_artist_id = None
            track_bytes = self.track_redis.get(cand_id)
            if track_bytes is not None:
                track = self.catalog.from_bytes(track_bytes)
                cand_artist_id = track.artist_id

            penalty = history_artist_counts.get(cand_artist_id, 0) * ARTIST_PENALTY_COEFF

            final_score = SEMANTIC_WEIGHT * semantic_score + HSTU_RANK_WEIGHT * hstu_rank_score - penalty

            if final_score > max_score:
                max_score = final_score
                best_track = cand_id

        return best_track if best_track is not None else candidates[0]

    def _select_anchor(self, history, prev_track):
        track_time = defaultdict(float)
        for track, listened_time in history:
            track_time[track] += listened_time

        if not track_time:
            return prev_track

        anchors = list(track_time.keys())
        weights = [track_time[t] for t in anchors]

        while anchors:
            chosen = random.choices(anchors, weights=weights, k=1)[0]
            if chosen in self.track_embeddings:
                return chosen

            idx = anchors.index(chosen)
            anchors.pop(idx)
            weights.pop(idx)

        return prev_track

    def _load_user_history(self, user):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)

        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history