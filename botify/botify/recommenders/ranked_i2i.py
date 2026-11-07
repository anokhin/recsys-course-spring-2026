import pickle
import random
from typing import List, Tuple

import numpy as np

from .recommender import Recommender


class RankedI2IRecommender(Recommender):
    def __init__(
        self,
        recommendations_redis,
        history_redis,
        catalog,
        fallback,
        context_emb: np.ndarray,
        track_emb: np.ndarray,
        temperature: float = 0.08,
    ):
        self.recommendations_redis = recommendations_redis
        self.history_redis = history_redis
        self.catalog = catalog
        self.fallback = fallback
        self.context_emb = context_emb
        self.track_emb = track_emb
        self.temperature = temperature

    def _load_history(self, user: int) -> List[Tuple[int, float]]:
        raw = self.history_redis.get(user)
        if raw is None:
            return []
        return pickle.loads(raw)

    def _save_history(self, user: int, history: List[Tuple[int, float]]) -> None:
        self.history_redis.set(user, pickle.dumps(history[-200:]))

    def _sample_context(self, history: List[Tuple[int, float]], prev_track: int) -> int:
        if not history:
            return prev_track
        recent = history[-20:]
        tracks, weights = zip(*recent)
        safe_weights = [max(float(w), 1e-6) ** 1.5 for w in weights]
        return random.choices(list(tracks), weights=safe_weights, k=1)[0]

    def _score(self, context: int, candidate: int) -> float:
        if (
            context < 0
            or candidate < 0
            or context >= len(self.context_emb)
            or candidate >= len(self.track_emb)
        ):
            return -1e9
        return float(np.dot(self.context_emb[context], self.track_emb[candidate]))

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_history(user)
        history.append((int(prev_track), float(prev_track_time)))
        self._save_history(user, history)

        context = self._sample_context(history, prev_track)
        raw = self.recommendations_redis.get(context)
        if raw is None:
            raw = self.recommendations_redis.get(prev_track)
        if raw is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        neighbors = list(self.catalog.from_bytes(raw))
        heard = {track for track, _ in history}
        candidates = [item for item in neighbors if item not in heard]
        if not candidates:
            candidates = [item for item in neighbors if item != prev_track]
        if not candidates:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        scores = np.array([self._score(context, c) for c in candidates], dtype=np.float64)
        scores = scores - scores.max()
        probs = np.exp(scores / max(self.temperature, 1e-6))
        probs = 0.85 * (probs / probs.sum()) + 0.15 / len(candidates)
        probs = probs / probs.sum()
        return int(np.random.choice(candidates, p=probs))
