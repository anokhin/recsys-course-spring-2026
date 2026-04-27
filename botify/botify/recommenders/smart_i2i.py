import json
import pickle
import random
from collections import defaultdict
from .recommender import Recommender


class SmartI2IRecommender(Recommender):
    """
    Умный I2I рекомендер, который:
    - Учитывает скипнутые треки как негативный сигнал
    - Взвешивает якоря по времени И негативным сигналам
    - Обеспечивает разнообразие рекомендаций
    """
    
    def __init__(self, listen_history_redis, i2i_redis, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback = fallback_recommender
        self.track_cache = {}
    
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
    
    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)
        
        if not history:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        
        # Разделяем на "хорошие" (>0.7) и "плохие" (<0.3) треки
        good_tracks = []
        bad_tracks = set()
        
        for track, listen_time in history:
            if listen_time >= 0.7:
                good_tracks.append((track, listen_time))
            elif listen_time < 0.3:
                bad_tracks.add(track)
        
        # Если нет хороших треков — fallback
        if not good_tracks:
            good_tracks = [(track, time) for track, time in history if time >= 0.3]
        
        if not good_tracks:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        
        # Веса: чем дольше слушал, тем больше вес якоря
        weights = [min(time, 1.0) for _, time in good_tracks]
        
        # Пробуем разные якоря (до 5 попыток)
        anchors = list(good_tracks)
        max_attempts = min(5, len(anchors))
        
        best_candidate = None
        best_score = -1
        
        for _ in range(max_attempts):
            if not anchors:
                break
            
            # Выбираем якорь по весу
            anchor_idx = random.choices(range(len(anchors)), weights=weights[:len(anchors)], k=1)[0]
            anchor_track, anchor_time = anchors[anchor_idx]
            
            # Получаем рекомендации для якоря
            candidates = self._get_recommendations(anchor_track)
            if not candidates:
                anchors.pop(anchor_idx)
                weights.pop(anchor_idx)
                continue
            
            # Оцениваем каждого кандидата
            for candidate in candidates:
                if candidate in seen_tracks:
                    continue
                
                score = self._score_candidate(candidate, anchor_track, anchor_time, bad_tracks)
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            # Убираем использованный якорь для разнообразия
            anchors.pop(anchor_idx)
            weights.pop(anchor_idx)
        
        if best_candidate is not None:
            return best_candidate
        
        return self.fallback.recommend_next(user, prev_track, prev_track_time)
    
    def _get_recommendations(self, track_id: int):
        """Получает I2I рекомендации из Redis."""
        if track_id in self.track_cache:
            return self.track_cache[track_id]
        
        data = self.i2i_redis.get(track_id)
        if data is None:
            return None
        
        recommendations = pickle.loads(data)
        recs = [int(t) for t in recommendations]
        self.track_cache[track_id] = recs
        return recs
    
    def _score_candidate(self, candidate, anchor_track, anchor_time, bad_tracks):
        """
        Оценивает кандидата:
        - Бонус за хороший якорь
        - Штраф если кандидат похож на скипнутые треки
        """
        score = anchor_time  # базовый вес от времени прослушивания якоря
        
        # Проверяем, не похож ли кандидат на скипнутые треки
        candidate_recs = self._get_recommendations(candidate)
        if candidate_recs:
            # Если среди рекомендаций кандидата много скипнутых треков — штраф
            overlap = len(set(candidate_recs[:5]) & bad_tracks)
            score -= overlap * 0.3
        
        # Бонус за похожесть на якорь (первые в списке — самые похожие)
        if candidate_recs and anchor_track in candidate_recs[:3]:
            score += 0.2
        
        return max(score, 0.1)
