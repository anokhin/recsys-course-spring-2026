"""Session-aware reranker built on learned item embeddings.

The recommender combines candidates retrieved from two pre-trained item-to-
item models (SasRec and LightFM) and reranks them in the embedding space
produced by ``botify.ml.train``. Scoring is deterministic and considers
both how well a candidate matches the current session intent and how
much it diversifies the user's recent artist exposure.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from botify.ml.artifacts import MLArtifacts
from .recommender import Recommender

HISTORY_DECAY = 0.85
SESSION_TIME_FLOOR = 0.05
CANDIDATES_PER_ANCHOR = 30
MAX_ANCHORS = 5
ARTIST_PENALTY = 0.18
SOURCE_AGREEMENT_BONUS = 0.04


class MLReranker(Recommender):
    """Reranks SasRec/LightFM candidates with learned item embeddings."""

    def __init__(
        self,
        listen_history_redis,
        sasrec_redis,
        lightfm_redis,
        artifacts: MLArtifacts,
        fallback: Recommender,
        history_limit: int = 10,
    ) -> None:
        self._history_redis = listen_history_redis
        self._sasrec_redis = sasrec_redis
        self._lightfm_redis = lightfm_redis
        self._artifacts = artifacts
        self._fallback = fallback
        self._history_limit = history_limit

    def recommend_next(
        self, user: int, prev_track: int, prev_track_time: float
    ) -> int:
        history = self._load_history(user)
        if not history:
            return self._fallback.recommend_next(user, prev_track, prev_track_time)

        seen = {track for track, _ in history}
        anchors = self._select_anchors(history)
        if not anchors:
            return self._fallback.recommend_next(user, prev_track, prev_track_time)

        sasrec_neighbors = self._fetch_neighbors(self._sasrec_redis, anchors)
        lightfm_neighbors = self._fetch_neighbors(self._lightfm_redis, anchors)
        candidates = self._build_candidate_pool(
            sasrec_neighbors, lightfm_neighbors, seen
        )
        if not candidates:
            return self._fallback.recommend_next(user, prev_track, prev_track_time)

        intent = self._build_intent_vector(history)
        if intent is None:
            return self._fallback.recommend_next(user, prev_track, prev_track_time)

        artist_counts = self._recent_artist_counts(history)
        ranked = self._rank_candidates(
            candidates, intent, artist_counts, sasrec_neighbors, lightfm_neighbors
        )
        if ranked is None:
            return self._fallback.recommend_next(user, prev_track, prev_track_time)
        return int(ranked)

    def _load_history(self, user: int) -> List[Tuple[int, float]]:
        key = f"user:{user}:listens"
        raw_entries = self._history_redis.lrange(key, 0, self._history_limit - 1)
        history: List[Tuple[int, float]] = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _select_anchors(
        self, history: Sequence[Tuple[int, float]]
    ) -> List[int]:
        weights: Dict[int, float] = {}
        for position, (track, played) in enumerate(history):
            weight = max(played, SESSION_TIME_FLOOR) * (HISTORY_DECAY ** position)
            weights[track] = weights.get(track, 0.0) + weight
        ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        return [track for track, _ in ranked[:MAX_ANCHORS]]

    def _fetch_neighbors(
        self, redis_client, anchors: Sequence[int]
    ) -> Dict[int, List[int]]:
        result: Dict[int, List[int]] = {}
        for anchor in anchors:
            payload = redis_client.get(anchor)
            if payload is None:
                continue
            try:
                neighbors = pickle.loads(payload)
            except (pickle.UnpicklingError, EOFError):
                continue
            result[anchor] = [int(track) for track in neighbors][
                :CANDIDATES_PER_ANCHOR
            ]
        return result

    @staticmethod
    def _build_candidate_pool(
        sasrec: Dict[int, List[int]],
        lightfm: Dict[int, List[int]],
        seen: set,
    ) -> List[int]:
        pool = set()
        for neighbors in sasrec.values():
            pool.update(neighbors)
        for neighbors in lightfm.values():
            pool.update(neighbors)
        return [track for track in pool if track not in seen]

    def _build_intent_vector(
        self, history: Sequence[Tuple[int, float]]
    ) -> Optional[np.ndarray]:
        embeddings = self._artifacts.embeddings
        accumulator = np.zeros(embeddings.dim, dtype=np.float32)
        total_weight = 0.0
        for position, (track, played) in enumerate(history):
            vector = embeddings.get(track)
            if vector is None:
                continue
            weight = max(played, SESSION_TIME_FLOOR) * (HISTORY_DECAY ** position)
            accumulator += vector * weight
            total_weight += weight
        if total_weight == 0.0:
            return None
        norm = float(np.linalg.norm(accumulator))
        if norm == 0.0:
            return None
        return accumulator / norm

    def _recent_artist_counts(
        self, history: Sequence[Tuple[int, float]]
    ) -> Counter:
        counts: Counter = Counter()
        for track, _ in history:
            artist = self._artifacts.artist_by_track.get(int(track))
            if artist is not None:
                counts[artist] += 1
        return counts

    def _rank_candidates(
        self,
        candidates: Iterable[int],
        intent: np.ndarray,
        artist_counts: Counter,
        sasrec: Dict[int, List[int]],
        lightfm: Dict[int, List[int]],
    ) -> Optional[int]:
        embeddings = self._artifacts.embeddings
        sasrec_set = {track for neighbors in sasrec.values() for track in neighbors}
        lightfm_set = {track for neighbors in lightfm.values() for track in neighbors}

        best_track: Optional[int] = None
        best_score = -np.inf
        for candidate in candidates:
            vector = embeddings.get(candidate)
            if vector is None:
                continue
            similarity = float(np.dot(intent, vector))
            artist = self._artifacts.artist_by_track.get(int(candidate))
            artist_count = artist_counts[artist] if artist is not None else 0
            agreement = (
                SOURCE_AGREEMENT_BONUS
                if candidate in sasrec_set and candidate in lightfm_set
                else 0.0
            )
            score = similarity - ARTIST_PENALTY * artist_count + agreement
            if score > best_score:
                best_score = score
                best_track = int(candidate)
        return best_track
