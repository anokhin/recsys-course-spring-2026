import json
import pickle
from collections import defaultdict
from typing import Iterable, List, Set, Tuple

from .recommender import Recommender


class HSTUHybridRecommender(Recommender):

    HSTU_WEIGHT = 0.2
    SASREC_WEIGHT = 1.0
    LIGHTFM_WEIGHT = 0.0
    RRF_K = 60

    MAX_HISTORY_ANCHORS = 1
    MAX_RECS_PER_SOURCE = 20

    def __init__(
        self,
        listen_history_redis,
        hstu_redis,
        sasrec_redis,
        lightfm_redis,
        fallback_recommender,
    ):
        self.listen_history_redis = listen_history_redis
        self.hstu_redis = hstu_redis
        self.sasrec_redis = sasrec_redis
        self.lightfm_redis = lightfm_redis
        self.fallback_recommender = fallback_recommender

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = {track for track, _ in history}
        seen_tracks.add(int(prev_track))

        scores = defaultdict(float)

        hstu_recs = self._load_recs(self.hstu_redis, user)
        self._add_ranked_candidates(
            scores=scores,
            candidates=hstu_recs,
            seen_tracks=seen_tracks,
            source_weight=self.HSTU_WEIGHT,
        )

        anchor_weights = self._get_history_tracks(history)
        for anchor, anchor_weight in anchor_weights[: self.MAX_HISTORY_ANCHORS]:
            sasrec_recs = self._load_recs(self.sasrec_redis, anchor)
            self._add_ranked_candidates(
                scores=scores,
                candidates=sasrec_recs,
                seen_tracks=seen_tracks,
                source_weight=self.SASREC_WEIGHT * anchor_weight,
            )

            lightfm_recs = self._load_recs(self.lightfm_redis, anchor)
            self._add_ranked_candidates(
                scores=scores,
                candidates=lightfm_recs,
                seen_tracks=seen_tracks,
                source_weight=self.LIGHTFM_WEIGHT * anchor_weight,
            )

        if scores:
            return max(scores, key=lambda track: (scores[track], -track))

        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

    def _add_ranked_candidates(
        self,
        scores,
        candidates: Iterable[int],
        seen_tracks: Set[int],
        source_weight: float,
    ) -> None:
        if source_weight <= 0:
            return
        for rank, candidate in enumerate(candidates):
            if rank >= self.MAX_RECS_PER_SOURCE:
                break
            candidate = int(candidate)
            if candidate in seen_tracks:
                continue
            scores[candidate] += source_weight / (self.RRF_K + rank + 1)

    def _get_history_tracks(self, history: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        if not history:
            return []

        weights = defaultdict(float)
        for pos, (track, listened_time) in enumerate(history):
            listened_time = max(float(listened_time), 0.0)
            if listened_time < 0.25:
                continue
            recency = 1.0 / (1.0 + 0.25 * pos)
            weights[int(track)] += listened_time * recency

        return sorted(weights.items(), key=lambda item: item[1], reverse=True)

    def _load_user_history(self, user: int) -> List[Tuple[int, float]]:
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                entry = json.loads(raw)
                history.append((int(entry["track"]), float(entry["time"])))
            except Exception:
                continue
        return history

    def _load_recs(self, redis_conn, key: int) -> List[int]:
        data = redis_conn.get(int(key))
        if data is None:
            return []
        try:
            return [int(track) for track in pickle.loads(data)]
        except Exception:
            return []
