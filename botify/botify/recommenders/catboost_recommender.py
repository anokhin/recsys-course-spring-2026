import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from botify.recommenders import Recommender
from botify.tracks import TracksCatalog
from botify.artists import ArtistsCatalog
from botify.users import UsersCatalog


class CatBoostRecommender(Recommender):
    """
    ML-рекомендер на базе CatBoostRanker.
    Использует эмбеддинги пользователей и треков, а также контекстные фичи.
    """

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

        # Загружаем модель
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models" / "cb_ranker.cbm"
        else:
            model_path = Path(model_path)

        if model_path.exists():
            import catboost
            self.model = catboost.CatBoostRanker()
            self.model.load_model(str(model_path))
            print(f"CatBoost модель загружена из {model_path}")
        else:
            print(f"ВНИМАНИЕ: Модель не найдена по пути {model_path}. Использую fallback.")
            self.model = None

        # Загружаем эмбеддинги треков
        embeddings_path = Path(__file__).parent.parent.parent / "sim" / "data" / "embeddings.npy"
        self.track_embeddings = np.load(embeddings_path)

        # Кэш для эмбеддингов пользователей
        self.user_embeddings_cache: Dict[int, np.ndarray] = {}

    def recommend(self, user_id: int, context: Dict[str, Any], n: int) -> List[int]:
        """
        Возвращает n рекомендованных треков для пользователя.
        """
        # Получаем эмбеддинг пользователя
        user_embedding = self._get_user_embedding(user_id, context)

        # Если модели нет — fallback на популярные треки
        if self.model is None:
            return self._fallback_recommend(context, n)

        # Кандидаты: все треки, кроме тех, что уже в истории
        history = context.get("tracks_history", [])
        candidate_tracks = [t for t in self.tracks_catalog.track_ids if t not in history]

        if not candidate_tracks:
            return []

        # Готовим фичи для CatBoost
        features = []
        for track_id in candidate_tracks:
            track_idx = self.tracks_catalog.get_track_index(track_id)
            if track_idx is None or track_idx >= len(self.track_embeddings):
                continue

            track_emb = self.track_embeddings[track_idx]

            # Фичи:
            # 1. Косинусное сходство user_emb и track_emb
            similarity = np.dot(user_embedding, track_emb) / (
                np.linalg.norm(user_embedding) * np.linalg.norm(track_emb) + 1e-9
            )

            # 2. Длительность трека (нормализованная)
            track_info = self.tracks_catalog.get_track(track_id)
            duration_norm = track_info.get("duration", 180) / 300.0 if track_info else 0.6

            # 3. Популярность трека
            popularity = track_info.get("popularity", 0.5) if track_info else 0.5

            # 4. Признак: трек уже был в истории? (всегда 0, т.к. отфильтровали)
            in_history = 0.0

            # 5. Время с последнего прослушивания (если есть в контексте)
            time_since_last = context.get("time_since_last_track", 300) / 300.0

            features.append([
                similarity,
                duration_norm,
                popularity,
                in_history,
                time_since_last,
            ])

        if not features:
            return self._fallback_recommend(context, n)

        # Предсказываем скоры
        scores = self.model.predict(features)

        # Сортируем и возвращаем топ-n
        track_score_pairs = list(zip(candidate_tracks[:len(scores)], scores))
        track_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return [track_id for track_id, _ in track_score_pairs[:n]]

    def _get_user_embedding(self, user_id: int, context: Dict[str, Any]) -> np.ndarray:
        """Вычисляет эмбеддинг пользователя как среднее эмбеддингов прослушанных треков."""
        if user_id in self.user_embeddings_cache:
            return self.user_embeddings_cache[user_id]

        history = context.get("tracks_history", [])
        if not history:
            # Новый пользователь — нулевой вектор
            emb_dim = self.track_embeddings.shape[1]
            return np.zeros(emb_dim)

        embeddings = []
        for track_id in history:
            track_idx = self.tracks_catalog.get_track_index(track_id)
            if track_idx is not None and track_idx < len(self.track_embeddings):
                embeddings.append(self.track_embeddings[track_idx])

        if not embeddings:
            emb_dim = self.track_embeddings.shape[1]
            return np.zeros(emb_dim)

        user_emb = np.mean(embeddings, axis=0)
        self.user_embeddings_cache[user_id] = user_emb
        return user_emb

    def _fallback_recommend(self, context: Dict[str, Any], n: int) -> List[int]:
        """Fallback: популярные треки, не из истории."""
        history = context.get("tracks_history", [])
        popular = [
            t for t in self.tracks_catalog.top_tracks[:100]
            if t not in history
        ]
        return popular[:n]