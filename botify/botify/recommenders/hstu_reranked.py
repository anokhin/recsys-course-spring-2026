import json
import pickle

from .recommender import Recommender


class HSTUReranked(Recommender):
    def __init__(self, listen_history_redis, hstu_redis, sasrec_redis, catalog, fallback):
        self.listen_history_redis = listen_history_redis
        self.hstu_redis = hstu_redis
        self.sasrec_redis = sasrec_redis
        self.catalog = catalog
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        hstu_data = self.hstu_redis.get(user)
        if hstu_data is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        hstu_candidates = list(self.catalog.from_bytes(hstu_data))
        seen_tracks = self._get_seen_tracks(user)
        sasrec_scores = self._get_sasrec_scores(prev_track)

        n = len(hstu_candidates)
        best_track = None
        best_score = (-1, -1)

        for rank, track in enumerate(hstu_candidates):
            if track in seen_tracks:
                continue
            score = (sasrec_scores.get(track, 0), n - rank)
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

    def _get_sasrec_scores(self, track: int) -> dict:
        data = self.sasrec_redis.get(track)
        if data is None:
            return {}
        recommendations = pickle.loads(data)
        return {int(t): len(recommendations) - i for i, t in enumerate(recommendations)}
