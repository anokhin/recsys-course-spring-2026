import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from catboost import CatBoostRegressor

from .recommender import Recommender


class CatBoostRecommender(Recommender):
    """ML-рекомендер на CatBoostRegressor с 9 признаками."""

    def __init__(self, tracks_catalog, artists_catalog, users_catalog):
        self.tracks_catalog = tracks_catalog
        self.track_ids = list(range(len(tracks_catalog.tracks)))
        
        # Загружаем модель
        model_path = Path(__file__).parent.parent.parent / "models" / "cb_regressor.cbm"
        
        if model_path.exists():
            self.model = CatBoostRegressor()
            self.model.load_model(str(model_path))
            print(f"[CatBoost] Модель загружена из {model_path}")
        else:
            print(f"[CatBoost] Модель не найдена по пути {model_path}")
            self.model = None
        
        # Загружаем эмбеддинги
        embeddings_path = Path(__file__).parent.parent.parent / "sim" / "data" / "embeddings.npy"
        self.track_embeddings = np.load(embeddings_path)
        
        # История пользователей и rewards
        self.user_history: Dict[int, List[int]] = defaultdict(list)
        self.user_rewards: Dict[int, List[float]] = defaultdict(list)
        
        print(f"[CatBoost] Загружено {len(self.track_ids)} треков")

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        # Сохраняем историю
        if prev_track >= 0:
            self.user_history[user].append(prev_track)
            self.user_rewards[user].append(prev_track_time)
        
        history = self.user_history[user]
        rewards = self.user_rewards[user]
        
        # Если нет модели или мало истории — fallback
        if self.model is None or len(history) < 2:
            return self._fallback_recommend(history)
        
        last_reward = rewards[-1] if rewards else 0.5
        
        # Эмбеддинг пользователя
        hist_embs = [self.track_embeddings[t] for t in history if t < len(self.track_embeddings)]
        user_emb = np.mean(hist_embs, axis=0) if hist_embs else np.zeros(self.track_embeddings.shape[1])
        
        # Статистики
        unique_tracks = len(set(history))
        diversity = unique_tracks / len(history) if history else 0
        
        if len(hist_embs) >= 2:
            hist_sims = []
            for j in range(len(hist_embs)-1):
                sim = np.dot(hist_embs[j], hist_embs[j+1]) / (
                    np.linalg.norm(hist_embs[j]) * np.linalg.norm(hist_embs[j+1]) + 1e-9
                )
                hist_sims.append(sim)
            avg_hist_sim = np.mean(hist_sims)
        else:
            avg_hist_sim = 0
        
        hist_std = np.std(hist_embs, axis=0).mean() if len(hist_embs) > 1 else 0
        
        # Кандидаты
        candidates = [t for t in self.track_ids if t not in history]
        if not candidates:
            return self._fallback_recommend(history)
        
        # Ограничиваем кандидатов
        if len(candidates) > 500:
            last_emb = self.track_embeddings[history[-1]]
            sims = np.dot(self.track_embeddings, last_emb)
            sims = sims / (np.linalg.norm(self.track_embeddings, axis=1) * np.linalg.norm(last_emb) + 1e-9)
            for t in history:
                if t < len(sims):
                    sims[t] = -1
            candidates = [int(i) for i in np.argsort(sims)[-500:] if i in candidates]
        
        # Собираем фичи
        features = []
        for track_id in candidates:
            if track_id >= len(self.track_embeddings):
                continue
            
            track_emb = self.track_embeddings[track_id]
            
            similarity = np.dot(user_emb, track_emb) / (
                np.linalg.norm(user_emb) * np.linalg.norm(track_emb) + 1e-9
            )
            
            feat = [
                similarity,
                len(history) / 50.0,
                last_reward,
                float(len(history)) / 50.0,
                np.linalg.norm(user_emb),
                np.linalg.norm(track_emb),
                diversity,
                avg_hist_sim,
                hist_std,
            ]
            features.append(feat)
        
        if not features:
            return self._fallback_recommend(history)
        
        scores = self.model.predict(np.array(features, dtype=np.float32))
        best_idx = np.argmax(scores)
        
        return candidates[best_idx]

    def _fallback_recommend(self, history: List[int]) -> int:
        candidates = [t for t in self.track_ids if t not in history]
        if candidates:
            return int(np.random.choice(candidates))
        return int(np.random.choice(self.track_ids))