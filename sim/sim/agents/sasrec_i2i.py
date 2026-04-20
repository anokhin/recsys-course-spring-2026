import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from sim.agents.base import Recommender


class SasRecI2IRecommender(Recommender):
    """Item-to-Item рекомендер на основе эмбеддингов."""

    def __init__(self, action_space):
        self.action_space = action_space
        
        # Загружаем треки (JSON Lines формат)
        tracks_path = Path(__file__).parent.parent.parent / "data" / "tracks.json"
        self.tracks = []
        with open(tracks_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.tracks.append(json.loads(line))
        
        self.track_ids = list(range(len(self.tracks)))
        print(f"[SasRec] Загружено {len(self.tracks)} треков")
        
        # Загружаем эмбеддинги
        embeddings_path = Path(__file__).parent.parent.parent / "data" / "embeddings.npy"
        self.track_embeddings = np.load(embeddings_path)
        print(f"[SasRec] Эмбеддинги: {self.track_embeddings.shape}")
        
        # История сессии
        self.current_session: Dict[int, List[int]] = {}

    def recommend(self, observation: Dict[str, Any], reward: float, done: bool) -> int:
        user_id = observation.get("user", 0)
        
        if user_id not in self.current_session:
            self.current_session[user_id] = []
        
        if done:
            if user_id in self.current_session:
                del self.current_session[user_id]
            return np.random.choice(self.track_ids)
        
        last_track = observation.get("last_track")
        if last_track is not None and last_track >= 0:
            self.current_session[user_id].append(last_track)
        
        history = self.current_session[user_id]
        
        # I2I: берём последний трек и находим похожие
        if history:
            last = history[-1]
            if last < len(self.track_embeddings):
                last_emb = self.track_embeddings[last]
                
                # Считаем сходство со всеми треками
                similarities = np.dot(self.track_embeddings, last_emb)
                norm = np.linalg.norm(self.track_embeddings, axis=1) * np.linalg.norm(last_emb)
                similarities = similarities / (norm + 1e-9)
                
                # Исключаем уже прослушанные
                for t in history:
                    if t < len(similarities):
                        similarities[t] = -1
                
                best = np.argmax(similarities)
                return int(best)
        
        # Fallback: случайный трек
        candidates = [t for t in self.track_ids if t not in history]
        if candidates:
            return np.random.choice(candidates)
        return np.random.choice(self.track_ids)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass