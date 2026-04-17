import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .recommender import Recommender


class HybridSessionRecommender(Recommender):
    """
    Session-aware hybrid ranker.

    Offline artifacts contain:
      - low-rank graph embeddings learned from fused SasRec/LightFM graphs;
      - low-rank text embeddings learned from track metadata;
      - precomputed nearest neighbours in the hybrid space;
      - optional HSTU per-user candidates;
      - popularity priors.

    Online scoring is deterministic and uses only the current session history.
    """

    def __init__(self, listen_history_redis, fallback_recommender, artifact_path: str):
        self.listen_history_redis = listen_history_redis
        self.fallback_recommender = fallback_recommender
        self.artifact_path = Path(artifact_path)
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        with self.artifact_path.open("rb") as f:
            data = pickle.load(f)

        self.graph_emb = np.asarray(data["graph_emb"], dtype=np.float32)
        self.text_emb = np.asarray(data["text_emb"], dtype=np.float32)
        self.artist_ids = np.asarray(data["artist_ids"])
        self.hybrid_neighbors = data["hybrid_neighbors"]
        self.sasrec_neighbors = data["sasrec_neighbors"]
        self.lightfm_neighbors = data["lightfm_neighbors"]
        self.hstu_candidates = data.get("hstu_candidates", {})
        self.popularity = np.asarray(data["popularity"], dtype=np.float32)
        self.popular_tracks = list(map(int, data["popular_tracks"]))

        self.n_items = int(self.graph_emb.shape[0])
        self.graph_norm = self._normalize(self.graph_emb)
        self.text_norm = self._normalize(self.text_emb)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x / denom

    def _load_user_history(self, user: int) -> List[Tuple[int, float]]:
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            track = int(entry["track"])
            listened_time = float(entry["time"])
            if 0 <= track < self.n_items:
                history.append((track, max(0.0, min(1.0, listened_time))))
        return history

    def _session_profile(
        self, history: Sequence[Tuple[int, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        graph_profile = np.zeros(self.graph_norm.shape[1], dtype=np.float32)
        text_profile = np.zeros(self.text_norm.shape[1], dtype=np.float32)

        total_weight = 0.0
        for position, (track, listened_time) in enumerate(history[:5]):
            recency = 0.82 ** position
            quality = 0.15 + listened_time
            weight = recency * quality
            graph_profile += weight * self.graph_norm[track]
            text_profile += weight * self.text_norm[track]
            total_weight += weight

        if total_weight > 0:
            graph_profile /= total_weight
            text_profile /= total_weight

        graph_norm = np.linalg.norm(graph_profile) + 1e-8
        text_norm = np.linalg.norm(text_profile) + 1e-8
        return graph_profile / graph_norm, text_profile / text_norm

    def _candidate_bonus_map(
        self, user: int, history: Sequence[Tuple[int, float]]
    ) -> Dict[int, float]:
        bonus: Dict[int, float] = {}

        def add(track: int, value: float) -> None:
            if 0 <= track < self.n_items:
                bonus[track] = bonus.get(track, 0.0) + float(value)

        for position, (anchor, listened_time) in enumerate(history[:5]):
            recency = 0.82 ** position
            quality = 0.15 + listened_time
            anchor_weight = recency * quality

            for rank, track in enumerate(self.sasrec_neighbors[anchor][:10]):
                add(track, anchor_weight * 0.18 / (rank + 1))

            for rank, track in enumerate(self.lightfm_neighbors[anchor][:10]):
                add(track, anchor_weight * 0.13 / (rank + 1))

            for rank, track in enumerate(self.hybrid_neighbors[anchor][:15]):
                add(track, anchor_weight * 0.12 / (rank + 1))

        if user in self.hstu_candidates:
            for rank, track in enumerate(self.hstu_candidates[user][:40]):
                add(track, 0.08 / (rank + 1))

        for rank, track in enumerate(self.popular_tracks[:120]):
            add(track, 0.015 / (rank + 1))

        return bonus

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        if not history:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        seen_tracks = {track for track, _ in history}
        artist_counter = Counter(int(self.artist_ids[track]) for track, _ in history)
        last_track = history[0][0]
        last_artist = int(self.artist_ids[last_track])

        session_graph, session_text = self._session_profile(history)
        candidate_bonus = self._candidate_bonus_map(user, history)

        best_track = None
        best_score = -10**9

        for candidate, source_bonus in candidate_bonus.items():
            if candidate in seen_tracks:
                continue

            candidate_artist = int(self.artist_ids[candidate])

            graph_sim = float(np.dot(self.graph_norm[candidate], session_graph))
            text_sim = float(np.dot(self.text_norm[candidate], session_text))
            last_graph_sim = float(np.dot(self.graph_norm[candidate], self.graph_norm[last_track]))
            last_text_sim = float(np.dot(self.text_norm[candidate], self.text_norm[last_track]))

            artist_repeats = artist_counter[candidate_artist]
            artist_penalty = 0.42 * artist_repeats
            if candidate_artist == last_artist:
                artist_penalty += 0.18

            score = (
                0.78 * graph_sim
                + 0.22 * text_sim
                + 0.10 * last_graph_sim
                + 0.03 * last_text_sim
                + source_bonus
                + 0.04 * float(self.popularity[candidate])
                - artist_penalty
            )

            if score > best_score or (
                math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12)
                and (best_track is None or candidate < best_track)
            ):
                best_score = score
                best_track = candidate

        if best_track is not None:
            return int(best_track)

        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)
