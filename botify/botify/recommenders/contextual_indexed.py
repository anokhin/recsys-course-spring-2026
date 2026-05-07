import json
import random
from collections import defaultdict
from .recommender import Recommender


class ContextualIndexedRecommender(Recommender):
    """
    Динамический рекомендер, использующий улучшенные I2I из Redis.
    Выбирает из нескольких якорей истории, фильтрует просмотренное.
    """

    def __init__(self, listen_history_redis, i2i_redis, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback = fallback_recommender

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)

        if not history:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        # Собираем треки-якоря с весами по времени прослушивания
        track_time = defaultdict(float)
        for track, listened_time in history[-10:]:  # Последние 10 треков
            if listened_time > 0.3:  # Только "лайкнутые"
                track_time[track] += listened_time

        # Если нет лайкнутых — берём все
        if not track_time:
            for track, listened_time in history[-5:]:
                track_time[track] = listened_time

        anchors = list(track_time.keys())
        weights = [track_time[t] for t in anchors]

        # Перебираем якоря, пока не найдём кандидата
        tried_anchors = 0
        while anchors and tried_anchors < 5:
            anchor = random.choices(anchors, weights=weights, k=1)[0]
            candidate = self._get_best_candidate(anchor, seen_tracks)
            
            if candidate is not None:
                return candidate

            # Убираем этот якорь и пробуем следующий
            idx = anchors.index(anchor)
            anchors.pop(idx)
            weights.pop(idx)
            tried_anchors += 1

        # Если ничего не нашли — fallback
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

    def _get_best_candidate(self, anchor: int, seen_tracks):
        """Берёт топ-1 непрослушанный трек из улучшенных I2I для якоря"""
        import pickle
        data = self.i2i_redis.get(anchor)
        if data is None:
            return None

        try:
            recommendations = pickle.loads(data)
        except:
            return None

        for track in recommendations:
            candidate = int(track)
            if candidate not in seen_tracks:
                return candidate
        return None