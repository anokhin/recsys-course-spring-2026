import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from sim.agents.base import Recommender
from sim.agents.sasrec_i2i import SasRecI2IRecommender
from catboost import CatBoostRegressor


class CatBoostRecommender(Recommender):
    """ML-рекомендер на CatBoost-регрессоре с 9 признаками."""

    def __init__(self, action_space):
        self.action_space = action_space
        self.sasrec = SasRecI2IRecommender(action_space)
        
        models_dir = Path(__file__).parent.parent.parent.parent / "models"
        
        # Загружаем модель
        self.model = CatBoostRegressor()
        self.model.load_model(str(models_dir / "cb_regressor.cbm"))
        
        with open(models_dir / "cb_reg_meta.json") as f:
            self.meta = json.load(f)
        
        self.track_embeddings = self.sasrec.track_embeddings
        self.track_ids = self.sasrec.track_ids
        
        self.current_session: Dict[int, List[int]] = {}
        self.session_rewards: Dict[int, List[float]] = {}
        
        print(f"[CatBoost] Модель загружена, RMSE: {self.meta['rmse']:.4f}")

    def recommend(self, observation: Dict[str, Any], reward: float, done: bool) -> int:
        user_id = observation.get("user", 0)
        
        if user_id not in self.current_session:
            self.current_session[user_id] = []
            self.session_rewards[user_id] = []
        
        if done:
            if user_id in self.current_session:
                del self.current_session[user_id]
            return 0
        
        last_track = observation.get("last_track")
        if last_track is not None and last_track >= 0:
            self.current_session[user_id].append(last_track)
            self.session_rewards[user_id].append(reward)
        
        history = self.current_session[user_id]
        
        if len(history) < 2:
            return self.sasrec.recommend(observation, reward, done)
        
        # Эмбеддинг пользователя
        hist_embs = [self.track_embeddings[t] for t in history if t < len(self.track_embeddings)]
        user_emb = np.mean(hist_embs, axis=0) if hist_embs else np.zeros(self.track_embeddings.shape[1])
        
        # Статистики истории
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
            return 0
        
        # Ограничиваем кандидатов для скорости
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
                reward,
                float(len(history)) / 50.0,
                np.linalg.norm(user_emb),
                np.linalg.norm(track_emb),
                diversity,
                avg_hist_sim,
                hist_std,
            ]
            features.append(feat)
        
        if not features:
            return self.sasrec.recommend(observation, reward, done)
        
        # Предсказываем
        scores = self.model.predict(np.array(features, dtype=np.float32))
        best_idx = np.argmax(scores)
        
        return candidates[best_idx]

    def __enter__(self):
        self.sasrec.__enter__()
        return self

    def __exit__(self, *args):
        self.sasrec.__exit__(*args)