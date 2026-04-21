import json
import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .recommender import Recommender

try:
    # feature-extraction helpers live in the training module; importing here guarantees
    # train-time and inference-time features stay identical.
    from script.train_reranker import build_candidate_features
except Exception:  # pragma: no cover - defensive for non-PYTHONPATH envs
    build_candidate_features = None


class LGBMReranker(Recommender):
    """
    Rerank top-K SasRec-I2I candidates with a pre-trained LightGBM booster.
    Deterministic: argmax by score, stable tie-break by candidate id.
    """

    def __init__(
        self,
        listen_history_redis,
        sasrec_redis,
        fallback,
        booster,
        feature_order: List[str],
        track_meta: Dict[int, dict],
        item2vec: Dict[int, np.ndarray],
        top_k: int = 50,
    ):
        self.listen_history_redis = listen_history_redis
        self.sasrec_redis = sasrec_redis
        self.fallback = fallback
        self.booster = booster
        self.feature_order = list(feature_order)
        self.track_meta = track_meta
        self.item2vec = item2vec
        self.top_k = int(top_k)
        assert build_candidate_features is not None, \
            "script.train_reranker must be importable"

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_history(user)
        if not history:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        seen = {t for t, _ in history}

        weights: Dict[int, float] = defaultdict(float)
        for track, t in history:
            weights[track] += t
        anchors = sorted(weights, key=weights.get, reverse=True)

        candidates: List[int] = []
        taken = set()
        for anchor in anchors:
            anchor_cands = self._candidates(anchor, seen, exclude=taken)
            for c in anchor_cands:
                candidates.append(c)
                taken.add(c)
                if len(candidates) >= self.top_k:
                    break
            if len(candidates) >= self.top_k:
                break

        if not candidates:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        X = np.zeros((len(candidates), len(self.feature_order)), dtype=np.float32)
        for i, cand in enumerate(candidates):
            feats = build_candidate_features(
                history=history,
                candidate=cand,
                candidate_rank=i,
                track_meta=self.track_meta,
                item2vec=self.item2vec,
            )
            for j, name in enumerate(self.feature_order):
                X[i, j] = feats[name]

        scores = np.asarray(self.booster.predict(X), dtype=np.float64)
        # argmax with stable tie-break by candidate id
        cands_arr = np.asarray(candidates, dtype=np.int64)
        best_idx = int(np.lexsort((cands_arr, -scores))[0])
        return int(candidates[best_idx])

    def _load_history(self, user: int):
        raw = self.listen_history_redis.lrange(f"user:{user}:listens", 0, -1)
        out = []
        for r in raw:
            if isinstance(r, bytes):
                r = r.decode("utf-8")
            entry = json.loads(r)
            out.append((int(entry["track"]), float(entry["time"])))
        return out

    def _candidates(self, anchor: int, seen, exclude):
        data = self.sasrec_redis.get(anchor)
        if data is None:
            return []
        recs = pickle.loads(data)
        return [int(t) for t in recs if int(t) not in seen and int(t) not in exclude]
