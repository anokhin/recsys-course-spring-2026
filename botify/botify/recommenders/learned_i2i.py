import json
import pickle

from .recommender import Recommender


class LearnedI2IRecommender(Recommender):
    def __init__(self, listen_history_redis, i2i_redis, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback_recommender = fallback_recommender

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        seen = self._seen_tracks(user)

        if prev_track is not None:
            candidates = self._load_candidates(prev_track)

            for track in candidates:
                candidate = int(track)
                if candidate != int(prev_track) and candidate not in seen:
                    return candidate

        fallback = self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        if fallback is not None:
            return int(fallback)

        return self._random_track()

    def _seen_tracks(self, user: int):
        key = f"user:{user}:listens"
        rows = self.listen_history_redis.lrange(key, 0, -1)

        seen = set()

        for raw in rows:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                row = json.loads(raw)
                if "track" in row:
                    seen.add(int(row["track"]))
                elif "track_id" in row:
                    seen.add(int(row["track_id"]))
            except Exception:
                pass

        return seen

    def _load_candidates(self, prev_track: int):
        raw = self.i2i_redis.get(int(prev_track))

        if raw is None:
            return []

        try:
            return pickle.loads(raw)
        except Exception:
            return []

    def _random_track(self):
        key = self.i2i_redis.randomkey()

        if isinstance(key, bytes):
            key = key.decode("utf-8")

        return int(key)
