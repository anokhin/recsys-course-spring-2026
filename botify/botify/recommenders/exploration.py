import json
import pickle
import random
from collections import defaultdict

from .recommender import Recommender


class ExplorationRecommender(Recommender):
    """
    ε-greedy wrapper over SasRec i2i top-K candidates used for *data collection only*.
    With prob 1-ε picks top-1 of the best anchor's candidates (= baseline SasRec-I2I
    behaviour). With prob ε samples uniformly from the top-K of that anchor.
    Seen tracks are always filtered out.
    """

    def __init__(
        self,
        listen_history_redis,
        i2i_redis,
        fallback,
        epsilon: float = 0.2,
        top_k: int = 50,
        seed: int = 0,
    ):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback = fallback
        self.epsilon = float(epsilon)
        self.top_k = int(top_k)
        self._rng = random.Random(seed)

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_history(user)
        seen = set(t for t, _ in history)

        if history:
            weights = defaultdict(float)
            for track, t in history:
                weights[track] += t
            anchors = list(weights)
            ws = [weights[a] for a in anchors]

            while anchors:
                anchor = self._rng.choices(anchors, weights=ws, k=1)[0]
                candidates = self._candidates(anchor, seen)
                if candidates:
                    if self._rng.random() < self.epsilon and len(candidates) > 1:
                        return self._rng.choice(candidates[: self.top_k])
                    return candidates[0]
                idx = anchors.index(anchor)
                anchors.pop(idx)
                ws.pop(idx)

        if self.fallback is None:
            raise RuntimeError("No fallback and no candidates")
        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    def _load_history(self, user: int):
        raw = self.listen_history_redis.lrange(f"user:{user}:listens", 0, -1)
        out = []
        for r in raw:
            if isinstance(r, bytes):
                r = r.decode("utf-8")
            entry = json.loads(r)
            out.append((int(entry["track"]), float(entry["time"])))
        return out

    def _candidates(self, anchor: int, seen):
        data = self.i2i_redis.get(anchor)
        if data is None:
            return []
        recs = pickle.loads(data)
        return [int(t) for t in recs if int(t) not in seen]
