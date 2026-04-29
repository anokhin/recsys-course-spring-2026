from .recommender import Recommender
import random
import json

class ContextualRanker(Recommender):
    def __init__(self, recommendations_redis, track_data, fallback, catalog, listen_history_redis):
        self.recommendations_redis = recommendations_redis
        self.track_data = track_data
        self.fallback = fallback
        self.catalog = catalog
        self.listen_history_redis = listen_history_redis # Добавили Redis для истории

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        # 1. Получаем кандидатов от SASREC
        recs_bytes = self.recommendations_redis.get(user)
        if recs_bytes is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        candidates = list(self.catalog.from_bytes(recs_bytes))
        
        # Если это самое начало сессии или кандидатов нет, возвращаем первый от SasRec
        if not candidates:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        
        # 2. Получаем историю прослушиваний пользователя из Redis
        user_history_key = f"user:{user}:listens"
        raw_history = self.listen_history_redis.lrange(user_history_key, 0, -1)
        
        # Декодируем историю
        listen_history = []
        for entry_bytes in raw_history:
            try:
                listen_history.append(json.loads(entry_bytes))
            except json.JSONDecodeError:
                continue # Пропускаем некорректные записи
        
        # Сохраняем предыдущий трек в истории (если он есть и дослушан)
        if prev_track is not None and prev_track_time is not None:
             # Имитируем, что prev_track только что добавлен в историю
             listen_history.insert(0, {"track": prev_track, "time": prev_track_time})


        # --- САМАЯ ГЛАВНАЯ СТРАТЕГИЯ: РЕПИТЫ ---
        # Проверяем, есть ли в кандидатах трек, который пользователь недавно слушал (и дослушал)
        # Если да, ставим его на первое место
        for history_item in listen_history:
            if history_item['time'] > 0.9: # Только если дослушал почти до конца
                if history_item['track'] in candidates:
                    return history_item['track'] # Возвращаем этот трек сразу!

        # --- СТРАТЕГИЯ 2: Sticky Artist (если предыдущий трек понравился) ---
        if prev_track is not None and prev_track_time > 0.8 and prev_track in self.track_data:
            prev_artist = self.track_data[prev_track].get("artist")
            if prev_artist:
                for t_id in candidates:
                    t_info = self.track_data.get(t_id)
                    # Ищем другой трек того же артиста (не тот же самый prev_track)
                    if t_info and t_info.get("artist") == prev_artist and t_id != prev_track:
                        return t_id # Ставим его на первое место
        
        # --- СТРАТЕГИЯ 3: Дефолтная ---
        # Если наши правила не сработали, просто возвращаем первый трек от SasRec
        return candidates[0]