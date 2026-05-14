import pickle
import json
from collections import Counter
from .recommender import Recommender

class SasRecVoting(Recommender):
    def __init__(self, listen_history_redis, i2i_redis, fallback, top_k=5):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback = fallback
        self.top_k = top_k

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)

        # Берём топ-K треков по времени прослушивания
        sorted_history = sorted(history, key=lambda x: x[1], reverse=True)
        anchors = [track for track, _ in sorted_history[:self.top_k]]

        # Voting: считаем сколько anchors рекомендуют каждый трек
        votes = Counter()
        for anchor in anchors:
            data = self.i2i_redis.get(anchor)
            if data is None:
                continue
            recommendations = pickle.loads(data)
            for track in recommendations[:20]:  # топ-20 рекомендаций каждого anchor
                candidate = int(track)
                if candidate not in seen_tracks:
                    votes[candidate] += 1

        if votes:
            return votes.most_common(1)[0][0]

        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    def _load_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history