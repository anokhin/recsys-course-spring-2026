import json
import pickle

from .recommender import Recommender


class RankedUserRecommender(Recommender):
    def __init__(
        self,
        listen_history_redis,
        user_recommendations_redis,
        fallback_recommender,
    ):
        self.listen_history_redis = listen_history_redis
        self.user_recommendations_redis = user_recommendations_redis
        self.fallback_recommender = fallback_recommender

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)
        recommendations = self._load_user_recommendations(user)

        if recommendations is not None:
            for track in recommendations:
                try:
                    candidate = int(track)
                except (TypeError, ValueError):
                    continue
                if candidate not in seen_tracks:
                    return candidate

        return self.fallback_recommender.recommend_next(
            user, prev_track, prev_track_time
        )

    def _load_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)

        history = []
        for raw in raw_entries:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                entry = json.loads(raw)
                history.append((int(entry["track"]), float(entry["time"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        return history

    def _load_user_recommendations(self, user: int):
        data = self.user_recommendations_redis.get(user)
        if data is None:
            return None

        try:
            recommendations = pickle.loads(data)
        except (pickle.UnpicklingError, EOFError, TypeError):
            return None

        if not isinstance(recommendations, list):
            return None

        return recommendations
