import json
import pickle

from botify.recommenders.recommender import Recommender


class MyRecommender(Recommender):
    def __init__(self, listen_history_redis, recommendations_redis, fallback):
        self.listen_history_redis = listen_history_redis
        self.recommendations_redis = recommendations_redis
        self.fallback = fallback

    def _loads(self, raw):
        if raw is None:
            return []
        try:
            return [int(x) for x in json.loads(raw)]
        except Exception:
            try:
                return [int(x) for x in pickle.loads(raw)]
            except Exception:
                return []

    def _seen(self, user):
        key = f"user:{user}:listens"
        raw_history = self.listen_history_redis.lrange(key, 0, -1)

        seen = set()

        for raw in raw_history:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                row = json.loads(raw)
                seen.add(int(row["track"]))
            except Exception:
                pass

        return seen

    def recommend_next(self, user, prev_track, prev_track_time):
        recs = self._loads(self.recommendations_redis.get(int(prev_track)))
        seen = self._seen(user)

        for rec in recs:
            if rec != int(prev_track) and rec not in seen:
                return rec

        return self.fallback.recommend_next(user, prev_track, prev_track_time)