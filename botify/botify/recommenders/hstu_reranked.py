import json
import pickle

from .recommender import Recommender

HSTU_WEIGHT = 0.01


class HSTUReranked(Recommender):
    def __init__(self, listen_history_redis, hstu_redis, sasrec_redis, catalog, fallback):
        self.listen_history_redis = listen_history_redis
        self.hstu_redis = hstu_redis
        self.sasrec_redis = sasrec_redis
        self.catalog = catalog
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        seen = self._get_seen_tracks(user)
        history = self._load_history(user)

        if not history:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        hstu_rank = {}
        hstu_size = 0
        hstu_data = self.hstu_redis.get(user)
        if hstu_data:
            candidates = list(self.catalog.from_bytes(hstu_data))
            hstu_size = len(candidates)
            hstu_rank = {t: i for i, t in enumerate(candidates)}

        best_track = None
        best_score = -1.0

        for anchor, listen_time in history:
            recs = self._get_recs(anchor)
            n = len(recs)
            for rank, track in enumerate(recs):
                if track in seen:
                    continue
                sasrec_score = listen_time * (n - rank)
                hstu_bonus = HSTU_WEIGHT * (hstu_size - hstu_rank[track]) if track in hstu_rank else 0.0
                score = sasrec_score + hstu_bonus
                if score > best_score:
                    best_score = score
                    best_track = track

        if best_track is not None:
            return best_track

        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    def _get_seen_tracks(self, user: int) -> set:
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        seen = set()
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            seen.add(int(entry["track"]))
        return seen

    def _load_history(self, user: int) -> list:
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _get_recs(self, anchor: int) -> list:
        data = self.sasrec_redis.get(anchor)
        if data is None:
            return []
        return [int(t) for t in pickle.loads(data)]
