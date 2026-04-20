import json
from pathlib import Path

import numpy as np

from .recommender import Recommender


class SessionSemanticRecommender(Recommender):
    def __init__(
        self,
        listen_history_redis,
        catalog,
        embeddings_path,
        fallback_recommender,
        i2i_redis=None,
        artist_penalty=0.0,
        min_weight=0.05,
        recent_history_limit=5,
        skip_time_threshold=0.2,
        max_semantic_anchors=4,
        max_i2i_anchors=3,
        semantic_candidate_limit=256,
        i2i_bonus=0.08,
        semantic_gate=0.18,
        min_margin=0.015,
    ):
        self.listen_history_redis = listen_history_redis
        self.catalog = catalog
        self.fallback_recommender = fallback_recommender
        self.i2i_redis = i2i_redis
        self.artist_penalty = artist_penalty
        self.min_weight = min_weight
        self.recent_history_limit = recent_history_limit
        self.skip_time_threshold = skip_time_threshold
        self.max_semantic_anchors = max_semantic_anchors
        self.max_i2i_anchors = max_i2i_anchors
        self.semantic_candidate_limit = semantic_candidate_limit
        self.i2i_bonus = i2i_bonus
        self.semantic_gate = semantic_gate
        self.min_margin = min_margin

        data = np.load(Path(embeddings_path))
        self.item_vectors = np.ascontiguousarray(data["vectors"].astype(np.float32))
        self.neighbors = np.ascontiguousarray(data["neighbors"].astype(np.int32))
        norms = np.linalg.norm(self.item_vectors, axis=1, keepdims=True) + 1e-8
        self.item_vectors_unit = self.item_vectors / norms

        if len(self.catalog.tracks) != self.item_vectors.shape[0]:
            raise ValueError("Track catalog size does not match semantic embeddings")

        artist_names = [track.artist for track in sorted(self.catalog.tracks, key=lambda x: x.track)]
        artist_to_id = {artist: idx for idx, artist in enumerate(sorted(set(artist_names)))}
        self.track_artist_ids = np.array(
            [artist_to_id[artist] for artist in artist_names],
            dtype=np.int32,
        )
        self.n_artists = len(artist_to_id)

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        if not history:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        recent_history = [
            (track, listened_time)
            for track, listened_time in history
            if listened_time >= self.skip_time_threshold
        ][: self.recent_history_limit]
        if not recent_history:
            recent_history = history[: self.recent_history_limit]

        history_tracks = np.array([track for track, _ in recent_history], dtype=np.int32)
        weights = np.array(
            [
                max(float(listened_time), self.min_weight) * (0.9 ** idx)
                for idx, (_, listened_time) in enumerate(recent_history)
            ],
            dtype=np.float32,
        )

        profile = self._estimate_session_profile(recent_history)
        profile /= np.linalg.norm(profile) + 1e-8

        seen_tracks = {track for track, _ in history}
        baseline_candidates = self._load_i2i_candidates(recent_history, seen_tracks)
        if len(seen_tracks) >= self.item_vectors.shape[0]:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        scores = self.item_vectors_unit @ profile
        if seen_tracks:
            seen_idx = np.fromiter(sorted(seen_tracks), dtype=np.int32)
            scores[seen_idx] = -np.inf

        if self.artist_penalty > 0.0:
            artist_counts = np.bincount(
                self.track_artist_ids[history_tracks],
                minlength=self.n_artists,
            )
            scores -= self.artist_penalty * artist_counts[self.track_artist_ids]

        baseline_bonus = np.zeros_like(scores, dtype=np.float32)
        baseline_bonus_map = {
            track: self.i2i_bonus * (len(baseline_candidates) - rank) / max(len(baseline_candidates), 1)
            for rank, track in enumerate(baseline_candidates)
        }
        if baseline_bonus_map:
            idx = np.fromiter(baseline_bonus_map.keys(), dtype=np.int32)
            baseline_bonus[idx] = np.fromiter(baseline_bonus_map.values(), dtype=np.float32)
            scores += baseline_bonus

        best_idx = int(np.argmax(scores))
        best_track = best_idx
        finite_scores = scores[np.isfinite(scores)]
        top_score = float(scores[best_idx])
        if finite_scores.size > 1:
            next_score = float(np.partition(finite_scores, -2)[-2])
            margin = top_score - next_score
        else:
            margin = top_score

        if top_score < self.semantic_gate and baseline_candidates:
            return int(baseline_candidates[0])
        if margin < self.min_margin and baseline_candidates:
            return int(baseline_candidates[0])

        return best_track

    def _estimate_session_profile(self, recent_history):
        tracks = np.array([track for track, _ in recent_history], dtype=np.int32)
        times = np.array([float(t) for _, t in recent_history], dtype=np.float32)
        X = self.item_vectors_unit[tracks]

        # A monotonic surrogate for hidden preference score in the simulator.
        y = np.clip(2.0 * times - 1.0, -1.0, 1.0)
        sample_weights = np.maximum(times, self.min_weight)

        prior = X[0]
        dim = X.shape[1]
        reg = 0.6
        A = reg * np.eye(dim, dtype=np.float32)
        b = reg * prior.astype(np.float32)
        for x, target, weight in zip(X, y, sample_weights):
            x = x.astype(np.float32)
            A += weight * np.outer(x, x)
            b += weight * target * x
        profile = np.linalg.solve(A, b)

        # Blend the fitted profile with the simple positive-preference centroid.
        positive = np.maximum(times, 0.05)
        centroid = (X * positive[:, None]).sum(axis=0) / positive.sum()
        return 0.65 * profile + 0.35 * centroid

    def _load_i2i_candidates(self, history, seen_tracks):
        if self.i2i_redis is None:
            return []

        ordered = []
        added = set()
        for anchor_track, _ in history[: self.max_i2i_anchors]:
            data = self.i2i_redis.get(anchor_track)
            if data is None:
                continue
            for candidate in self.catalog.from_bytes(data):
                candidate = int(candidate)
                if candidate in seen_tracks or candidate in added:
                    continue
                ordered.append(candidate)
                added.add(candidate)
        return ordered

    def _load_user_history(self, user: int):
        raw_entries = self.listen_history_redis.lrange(f"user:{user}:listens", 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history
