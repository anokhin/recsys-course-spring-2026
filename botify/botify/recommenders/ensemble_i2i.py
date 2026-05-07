import json
import pickle
from collections import defaultdict

from .recommender import Recommender


class EnsembleI2I(Recommender):
    def __init__(self, listen_history_redis, sasrec_redis, lfm_redis, fallback):
        self.listen_history_redis = listen_history_redis
        self.sasrec_redis = sasrec_redis
        self.lfm_redis = lfm_redis
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_history(user)
        if not history:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        seen_tracks = {track for track, _ in history}
        total_time = sum(t for _, t in history) or 1.0

        scores = defaultdict(float)
        for anchor, listen_time in history:
            weight = listen_time / total_time
            for redis_conn in (self.sasrec_redis, self.lfm_redis):
                recs = self._get_recs(redis_conn, anchor)
                n = len(recs)
                for rank, track in enumerate(recs):
                    scores[track] += weight * (n - rank)

        best = max(
            ((track, score) for track, score in scores.items() if track not in seen_tracks),
            key=lambda x: x[1],
            default=None,
        )
        if best:
            return best[0]

        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    def _load_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _get_recs(self, redis_conn, track: int) -> list:
        data = redis_conn.get(track)
        if data is None:
            return []
        return [int(t) for t in pickle.loads(data)]
