import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from .recommender import Recommender


class ALSRecommender(Recommender):
    """ML-рекомендер на ALS (матричная факторизация)."""

    def __init__(self, tracks_catalog, artists_catalog, users_catalog):
        self.tracks_catalog = tracks_catalog
        self.track_ids = list(range(len(tracks_catalog.tracks)))
        
        models_dir = Path(__file__).parent.parent.parent / "models"
        
        # Загружаем модель
        with open(models_dir / "als_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        
        with open(models_dir / "als_mappings.pkl", "rb") as f:
            mappings = pickle.load(f)
        
        self.user_ids = mappings["user_ids"]
        self.item_ids = mappings["item_ids"]
        self.id_to_item = {v: int(k) for k, v in self.item_ids.items()}
        
        self.user_history = defaultdict(list)
        
        print(f"[ALS] Загружено: {len(self.user_ids)} пользователей, {len(self.item_ids)} треков")

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        if prev_track >= 0:
            self.user_history[user].append(prev_track)
        
        history = self.user_history[user]
        
        # Если пользователь есть в модели
        if user in self.user_ids:
            user_idx = self.user_ids[user]
            
            # Предсказываем скоры для всех треков
            user_vector = self.model.user_factors[user_idx]
            scores = user_vector @ self.model.item_factors.T
            
            # Сортируем
            sorted_items = np.argsort(scores)[::-1]
            
            # Ищем первый трек не из истории
            for item_idx in sorted_items:
                track = self.id_to_item.get(item_idx)
                if track is not None and track not in history:
                    return track
        
        # Fallback: популярные треки не из истории
        popular = list(self.item_ids.keys())[:500]
        candidates = [int(t) for t in popular if int(t) not in history]
        if candidates:
            return int(np.random.choice(candidates[:10]))
        
        return int(np.random.choice(self.track_ids))

    def __exit__(self, *args):
        pass