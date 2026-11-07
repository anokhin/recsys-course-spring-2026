import pickle
import random
from typing import List, Tuple

from .recommender import Recommender


class I2IRecommender(Recommender):
    def __init__(self, recommendations_redis, history_redis, catalog, fallback):
        self.recommendations_redis = recommendations_redis
        self.history_redis = history_redis
        self.catalog = catalog
        self.fallback = fallback

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
        tracks, weights = zip(*history)
        safe_weights = [max(float(w), 1e-6) for w in weights]
        return random.choices(list(tracks), weights=safe_weights, k=1)[0]

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_history(user)
        history.append((int(prev_track), float(prev_track_time)))
        self._save_history(user, history)

        context = self._sample_context(history, prev_track)
        raw = self.recommendations_redis.get(context)
        if raw is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        neighbors = list(self.catalog.from_bytes(raw))
        heard = {track for track, _ in history}
        candidates = [item for item in neighbors if item not in heard]
        if not candidates:
            candidates = [item for item in neighbors if item != prev_track]
        if not candidates:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        random.shuffle(candidates)
        return int(candidates[0])
