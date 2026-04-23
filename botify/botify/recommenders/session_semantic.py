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
        lfm_i2i_redis=None,
        hstu_redis=None,
        artist_penalty=0.22,
        min_weight=0.05,
        recent_history_limit=6,
        skip_time_threshold=0.2,
        max_i2i_anchors=3,
        session_profile_weight=0.78,
        prototype_weight=0.22,
        hstu_prior_weight=0.0,
        negative_penalty=0.08,
        max_user_prototypes=6,
        prototype_match_threshold=0.58,
        max_semantic_anchors=4,
        semantic_neighbors_per_anchor=96,
        i2i_bonus=0.07,
        lfm_bonus=0.0,
        hstu_bonus=0.0,
        semantic_gate=0.14,
        min_margin=0.010,
    ):
        self.listen_history_redis = listen_history_redis
        self.catalog = catalog
        self.fallback_recommender = fallback_recommender
        self.i2i_redis = i2i_redis
        self.lfm_i2i_redis = lfm_i2i_redis
        self.hstu_redis = hstu_redis
        self.artist_penalty = artist_penalty
        self.min_weight = min_weight
        self.recent_history_limit = recent_history_limit
        self.skip_time_threshold = skip_time_threshold
        self.max_i2i_anchors = max_i2i_anchors
        self.session_profile_weight = session_profile_weight
        self.prototype_weight = prototype_weight
        self.hstu_prior_weight = hstu_prior_weight
        self.negative_penalty = negative_penalty
        self.max_user_prototypes = max_user_prototypes
        self.prototype_match_threshold = prototype_match_threshold
        self.max_semantic_anchors = max_semantic_anchors
        self.semantic_neighbors_per_anchor = semantic_neighbors_per_anchor
        self.i2i_bonus = i2i_bonus
        self.lfm_bonus = lfm_bonus
        self.hstu_bonus = hstu_bonus
        self.semantic_gate = semantic_gate
        self.min_margin = min_margin

        data = np.load(Path(embeddings_path))
        self.item_vectors = np.ascontiguousarray(data["vectors"].astype(np.float32))
        self.neighbors = np.ascontiguousarray(data["neighbors"].astype(np.int32))
        norms = np.linalg.norm(self.item_vectors, axis=1, keepdims=True) + 1e-8
        self.item_vectors_unit = self.item_vectors / norms

        if len(self.catalog.tracks) != self.item_vectors.shape[0]:
            raise ValueError("Track catalog size does not match semantic embeddings")

        artist_names = [
            track.artist
            for track in sorted(self.catalog.tracks, key=lambda item: item.track)
        ]
        artist_to_id = {
            artist: idx for idx, artist in enumerate(sorted(set(artist_names)))
        }
        self.track_artist_ids = np.array(
            [artist_to_id[artist] for artist in artist_names],
            dtype=np.int32,
        )
        self.n_artists = len(artist_to_id)

        self.active_sessions = {}
        self.user_prototypes = {}
        self.user_prototype_weights = {}

    def observe(self, user: int, track: int, listened_time: float):
        self.active_sessions.setdefault(user, []).append(
            (int(track), float(listened_time))
        )

    def finish_session(self, user: int):
        session_history = self.active_sessions.pop(user, None)
        if not session_history:
            return

        positive_history = [
            (track, listened_time)
            for track, listened_time in session_history
            if listened_time >= self.skip_time_threshold
        ]
        if not positive_history:
            return

        session_vector = self._weighted_centroid(positive_history, decay=0.95)
        if session_vector is None:
            return

        session_weight = float(
            sum(
                max(float(listened_time), self.min_weight)
                for _, listened_time in positive_history
            )
        )
        prototypes = self.user_prototypes.setdefault(user, [])
        prototype_weights = self.user_prototype_weights.setdefault(user, [])

        if prototypes:
            sims = np.asarray(
                [float(np.dot(proto, session_vector)) for proto in prototypes],
                dtype=np.float32,
            )
            best_idx = int(np.argmax(sims))
            if float(sims[best_idx]) >= self.prototype_match_threshold:
                old_proto = prototypes[best_idx]
                old_weight = float(prototype_weights[best_idx])
                blend = min(
                    session_weight / max(old_weight + session_weight, self.min_weight),
                    0.35,
                )
                prototypes[best_idx] = self._normalize(
                    (1.0 - blend) * old_proto + blend * session_vector
                )
                prototype_weights[best_idx] = min(
                    old_weight * 0.97 + session_weight,
                    24.0,
                )
                return

        if len(prototypes) < self.max_user_prototypes:
            prototypes.append(session_vector)
            prototype_weights.append(session_weight)
            return

        weakest_idx = int(np.argmin(np.asarray(prototype_weights, dtype=np.float32)))
        if session_weight >= 0.85 * float(prototype_weights[weakest_idx]):
            prototypes[weakest_idx] = session_vector
            prototype_weights[weakest_idx] = session_weight

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        session_history = list(self.active_sessions.get(user, []))
        if not session_history:
            session_history = list(reversed(self._load_user_history(user)))[
                : self.recent_history_limit
            ]
        if not session_history:
            return self.fallback_recommender.recommend_next(
                user, prev_track, prev_track_time
            )

        session_recent = session_history[-self.recent_history_limit :]
        positive_history = [
            (track, listened_time)
            for track, listened_time in session_recent
            if listened_time >= self.skip_time_threshold
        ]
        if not positive_history:
            positive_history = session_recent

        seen_tracks = {track for track, _ in session_history}
        if len(seen_tracks) >= self.item_vectors.shape[0]:
            return self.fallback_recommender.recommend_next(
                user, prev_track, prev_track_time
            )

        session_profile = self._estimate_session_profile(session_recent)
        prototype_prior = self._select_user_prototype(
            user, session_profile, positive_history
        )
        negative_profile = self._negative_profile(session_recent)

        hstu_candidates = self._load_user_candidates(
            self.hstu_redis, user, seen_tracks
        )
        hstu_prior = self._prior_from_candidates(hstu_candidates)

        profile = self._combine_profiles(
            session_profile, prototype_prior, hstu_prior
        )
        if profile is None:
            return self.fallback_recommender.recommend_next(
                user, prev_track, prev_track_time
            )

        semantic_candidates = self._load_semantic_candidates(
            positive_history, seen_tracks
        )
        source_bonus = {}
        sas_candidates = self._load_i2i_candidates(
            positive_history, seen_tracks, self.i2i_redis
        )
        lfm_candidates = self._load_i2i_candidates(
            positive_history, seen_tracks, self.lfm_i2i_redis
        )
        self._merge_bonus(source_bonus, sas_candidates, self.i2i_bonus)
        self._merge_bonus(source_bonus, lfm_candidates, self.lfm_bonus)
        self._merge_bonus(source_bonus, hstu_candidates, self.hstu_bonus)

        candidate_pool = self._merge_candidates(
            semantic_candidates,
            sas_candidates,
            lfm_candidates,
            hstu_candidates,
        )
        if not candidate_pool:
            return self.fallback_recommender.recommend_next(
                user, prev_track, prev_track_time
            )

        candidate_pool = np.asarray(candidate_pool, dtype=np.int32)
        candidate_vectors = self.item_vectors_unit[candidate_pool]
        scores = candidate_vectors @ profile
        if negative_profile is not None:
            scores -= self.negative_penalty * (candidate_vectors @ negative_profile)

        if self.artist_penalty > 0.0:
            session_artists = np.array(
                [self.track_artist_ids[int(track)] for track, _ in session_history],
                dtype=np.int32,
            )
            artist_counts = np.bincount(session_artists, minlength=self.n_artists)
            scores -= self.artist_penalty * artist_counts[
                self.track_artist_ids[candidate_pool]
            ]

        if source_bonus:
            for idx, track in enumerate(candidate_pool):
                scores[idx] += source_bonus.get(int(track), 0.0)

        best_idx = int(np.argmax(scores))
        recommendation = int(candidate_pool[best_idx])
        top_score = float(scores[best_idx])
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size > 1:
            next_score = float(np.partition(finite_scores, -2)[-2])
            margin = top_score - next_score
        else:
            margin = top_score

        fallback_track = sas_candidates[0] if sas_candidates else None
        if fallback_track is not None and top_score < self.semantic_gate:
            return int(fallback_track)
        if fallback_track is not None and margin < self.min_margin:
            return int(fallback_track)

        return recommendation

    def _estimate_session_profile(self, session_history):
        tracks = np.asarray([int(track) for track, _ in session_history], dtype=np.int32)
        times = np.asarray([float(t) for _, t in session_history], dtype=np.float32)
        vectors = self.item_vectors_unit[tracks]

        y = np.log(
            np.clip(times, 1e-3, 1.0 - 1e-3)
            / np.clip(1.0 - times, 1e-3, 1.0)
        )
        sample_weights = np.maximum(times, self.min_weight)

        prior = vectors[int(np.argmax(sample_weights))]
        dim = vectors.shape[1]
        reg = 0.55
        matrix = reg * np.eye(dim, dtype=np.float32)
        rhs = reg * prior.astype(np.float32)
        for vector, target, weight in zip(vectors, y, sample_weights):
            vector = vector.astype(np.float32)
            matrix += weight * np.outer(vector, vector)
            rhs += weight * target * vector
        profile = np.linalg.solve(matrix, rhs)

        positive_centroid = self._weighted_centroid(session_history, decay=0.97)
        if positive_centroid is None:
            return self._normalize(profile)
        return self._normalize(0.72 * profile + 0.28 * positive_centroid)

    def _select_user_prototype(self, user, session_profile, positive_history):
        prototypes = self.user_prototypes.get(user)
        if not prototypes:
            return None

        if session_profile is not None:
            anchor = session_profile
        else:
            anchor = self._weighted_centroid(positive_history, decay=0.98)
        if anchor is None:
            return None

        sims = np.asarray(
            [float(np.dot(proto, anchor)) for proto in prototypes],
            dtype=np.float32,
        )
        best_idx = int(np.argmax(sims))
        if float(sims[best_idx]) < 0.18:
            return None
        return prototypes[best_idx]

    def _negative_profile(self, session_history):
        negatives = [
            (track, listened_time)
            for track, listened_time in session_history
            if listened_time < self.skip_time_threshold
        ]
        if not negatives:
            return None

        vectors = []
        weights = []
        for idx, (track, listened_time) in enumerate(reversed(negatives[-4:])):
            vectors.append(self.item_vectors_unit[int(track)])
            weights.append(
                max(1.0 - float(listened_time), self.min_weight) * (0.9 ** idx)
            )
        if not weights or sum(weights) <= 0:
            return None
        centroid = np.average(
            np.asarray(vectors, dtype=np.float32),
            axis=0,
            weights=np.asarray(weights, dtype=np.float32),
        )
        return self._normalize(centroid)

    def _combine_profiles(self, session_profile, prototype_prior, hstu_prior):
        parts = []
        if session_profile is not None:
            parts.append(self.session_profile_weight * session_profile)
        if prototype_prior is not None:
            parts.append(self.prototype_weight * prototype_prior)
        if hstu_prior is not None:
            parts.append(self.hstu_prior_weight * hstu_prior)
        if not parts:
            return None
        return self._normalize(np.sum(parts, axis=0))

    def _weighted_centroid(self, history, decay):
        if not history:
            return None

        vectors = []
        weights = []
        for idx, (track, listened_time) in enumerate(reversed(history[-10:])):
            weight = max(float(listened_time), self.min_weight) * (decay ** idx)
            vectors.append(self.item_vectors_unit[int(track)])
            weights.append(weight)

        if not weights or sum(weights) <= 0:
            return None
        centroid = np.average(
            np.asarray(vectors, dtype=np.float32),
            axis=0,
            weights=np.asarray(weights, dtype=np.float32),
        )
        return self._normalize(centroid)

    def _load_i2i_candidates(self, history, seen_tracks, redis):
        if redis is None:
            return []

        ordered = []
        added = set()
        for anchor_track, _ in reversed(history[-self.max_i2i_anchors :]):
            data = redis.get(int(anchor_track))
            if data is None:
                continue
            for candidate in self.catalog.from_bytes(data):
                candidate = int(candidate)
                if candidate in seen_tracks or candidate in added:
                    continue
                ordered.append(candidate)
                added.add(candidate)
                if len(ordered) >= 32:
                    return ordered
        return ordered

    def _load_semantic_candidates(self, history, seen_tracks):
        ordered = []
        added = set()
        for anchor_track, _ in reversed(history[-self.max_semantic_anchors:]):
            for candidate in self.neighbors[int(anchor_track)][
                : self.semantic_neighbors_per_anchor
            ]:
                candidate = int(candidate)
                if candidate in seen_tracks or candidate in added:
                    continue
                ordered.append(candidate)
                added.add(candidate)
        return ordered

    def _merge_candidates(self, *candidate_lists):
        ordered = []
        added = set()
        for candidates in candidate_lists:
            for track in candidates:
                track = int(track)
                if track in added:
                    continue
                ordered.append(track)
                added.add(track)
        return ordered

    def _load_user_candidates(self, redis, user, seen_tracks):
        if redis is None:
            return []
        data = redis.get(user)
        if data is None:
            return []

        ordered = []
        for candidate in self.catalog.from_bytes(data):
            candidate = int(candidate)
            if candidate in seen_tracks:
                continue
            ordered.append(candidate)
            if len(ordered) >= 16:
                break
        return ordered

    def _prior_from_candidates(self, candidates):
        if not candidates:
            return None
        vectors = self.item_vectors_unit[np.asarray(candidates[:8], dtype=np.int32)]
        weights = np.asarray(
            [1.0 / (rank + 1) for rank in range(len(vectors))],
            dtype=np.float32,
        )
        centroid = np.average(vectors, axis=0, weights=weights)
        return self._normalize(centroid)

    def _merge_bonus(self, source_bonus, candidates, base_bonus):
        size = max(len(candidates), 1)
        for rank, track in enumerate(candidates):
            source_bonus[track] = (
                source_bonus.get(track, 0.0)
                + base_bonus * (size - rank) / size
            )

    def _load_user_history(self, user: int):
        raw_entries = self.listen_history_redis.lrange(f"user:{user}:listens", 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _normalize(self, vector):
        norm = np.linalg.norm(vector) + 1e-8
        return vector / norm
