import json
import pickle
import random
from collections import defaultdict
from .recommender import Recommender


class SmartI2IRecommender(Recommender):
    """
    Умный I2I рекомендер с улучшенным взвешиванием:
    - Учитывает скипнутые треки как негативный сигнал
    - Взвешивает якоря по времени И давности (recency)
    - Обеспечивает разнообразие через штраф за похожие треки в сессии
    - Использует глобальную статистику популярности треков
    """
    
    def __init__(self, listen_history_redis, i2i_redis, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback = fallback_recommender
        self.track_cache = {}
        self.global_popularity = defaultdict(int)
    
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
        
        # Обновляем глобальную популярность
        for track, listen_time in history:
            if listen_time >= 0.5:
                self.global_popularity[track] += 1
        
        # Разделяем на хорошие и плохие
        good_tracks = []
        bad_tracks = set()
        total_weight = 0
        
        for i, (track, listen_time) in enumerate(reversed(history)):
            recency = (i + 1) / len(history)  # более свежие — важнее
            weight = listen_time * recency
            
            if listen_time >= 0.7:
                good_tracks.append((track, weight))
                total_weight += weight
            elif listen_time < 0.3:
                bad_tracks.add(track)
        
        if not good_tracks:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        
        # Добавляем предыдущий трек как якорь если он не плохой
        if prev_track not in bad_tracks and prev_track not in seen_tracks:
            good_tracks.append((prev_track, 0.5))
        
        # Сортируем по весу (лучшие якоря первые)
        good_tracks.sort(key=lambda x: x[1], reverse=True)
        
        # Нормализуем веса
        weights = [w / max(total_weight, 0.1) for _, w in good_tracks]
        
        # Выбираем топ-5 якорей для разнообразия
        top_anchors = good_tracks[:5]
        top_weights = weights[:5]
        
        # Собираем кандидатов от всех топ-якорей с их позициями
        candidate_scores = defaultdict(float)
        
        for (anchor_track, anchor_weight), w in zip(top_anchors, top_weights):
            recs = self._get_recommendations(anchor_track)
            if recs is None:
                continue
            
            for position, candidate in enumerate(recs):
                if candidate in seen_tracks:
                    continue
                
                # Скоринг:
                score = 0.0
                
                # 1. Вес якоря × позиция в рекомендациях (первые лучше)
                position_bonus = 1.0 / (1 + position)  # 1.0, 0.5, 0.33, ...
                score += w * position_bonus * 2.0
                
                # 2. Бонус за популярность трека
                pop_bonus = min(self.global_popularity.get(candidate, 0) / 10.0, 0.3)
                score += pop_bonus
                
                # 3. Штраф за скипнутые треки в рекомендациях
                candidate_recs = self._get_recommendations(candidate)
                if candidate_recs:
                    skip_overlap = len(set(candidate_recs[:3]) & bad_tracks)
                    score -= skip_overlap * 0.3
                
                # 4. Штраф за дублирование в сессии (разнообразие)
                if candidate_recs:
                    session_overlap = len(set(candidate_recs[:5]) & seen_tracks)
                    score -= session_overlap * 0.1
                
                candidate_scores[candidate] += score
        
        # Выбираем лучшего кандидата
        if candidate_scores:
            best_candidate = max(candidate_scores, key=candidate_scores.get)
            return best_candidate
        
        # Fallback: пробуем просто топ-якоря по очереди
        for anchor_track, _ in top_anchors:
            recs = self._get_recommendations(anchor_track)
            if recs:
                for candidate in recs:
                    if candidate not in seen_tracks:
                        return candidate
        
        return self.fallback.recommend_next(user, prev_track, prev_track_time)
    
    def _get_recommendations(self, track_id: int):
        if track_id in self.track_cache:
            return self.track_cache[track_id]
        
        data = self.i2i_redis.get(track_id)
        if data is None:
            return None
        
        recommendations = pickle.loads(data)
        recs = [int(t) for t in recommendations]
        self.track_cache[track_id] = recs
        return recs
