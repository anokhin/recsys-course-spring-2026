import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from catboost import CatBoostRegressor

from botify.recommenders import Recommender
from botify.tracks import TracksCatalog
from botify.artists import ArtistsCatalog
from botify.users import UsersCatalog


class CatBoostRecommender(Recommender):
    """ML-рекомендер на CatBoostRegressor с 9 признаками."""

    def __init__(
        self,
        tracks_catalog: TracksCatalog,
        artists_catalog: ArtistsCatalog,
        users_catalog: UsersCatalog,
        model_path: Optional[str] = None,
    ):
        self.tracks_catalog = tracks_catalog
        self.artists_catalog = artists_catalog
        self.users_catalog = users_catalog

        # Путь к модели
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models" / "cb_regressor.cbm"
        else:
            model_path = Path(model_path)

        if model_path.exists():
            self.model = CatBoostRegressor()
            self.model.load_model(str(model_path))
            print(f"[CatBoost] Модель загружена из {model_path}")
        else:
            print(f"[CatBoost] ВНИМАНИЕ: Модель не найдена по пути {model_path}")
            self.model = None

        # Загружаем метаданные
        meta_path = Path(__file__).parent.parent.parent / "models" / "cb_reg_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
            print(f"[CatBoost] RMSE: {self.meta.get('rmse', 'N/A')}")
        else:
            self.meta = {}

        # Загружаем эмбеддинги треков
        embeddings_path = Path(__file__).parent.parent.parent / "sim" / "data" / "embeddings.npy"
        self.track_embeddings = np.load(embeddings_path)

        self.track_ids = list(range(len(self.tracks_catalog.tracks)))

    def recommend(self, user_id: int, context: Dict[str, Any], n: int) -> List[int]:
        history = context.get("tracks_history", [])
        last_reward = context.get("last_reward", 0.5)

        if len(history) < 2 or self.model is None:
            return self._fallback_recommend(context, n)

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
            return self._fallback_recommend(context, n)

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
            return self._fallback_recommend(context, n)

        scores = self.model.predict(np.array(features, dtype=np.float32))
        best_indices = np.argsort(scores)[::-1][:n]

        return [candidates[i] for i in best_indices]

    def _fallback_recommend(self, context: Dict[str, Any], n: int) -> List[int]:
        history = context.get("tracks_history", [])
        popular = [t for t in self.tracks_catalog.top_tracks[:100] if t not in history]
        return popular[:n]