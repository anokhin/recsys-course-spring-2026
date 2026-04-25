import random
from collections import Counter

class SimpleRanker:
    def __init__(self, action_space):
        self.action_space = action_space
        self.name = "SimpleRanker"
        self.history = []
        self.popularity = Counter()
        self.all_tracks = list(range(2000))
        
    def _get_track_id(self, observation):
        if observation is None:
            return None
        if isinstance(observation, dict):
            return observation.get('track_id', observation.get('id', None))
        return observation if isinstance(observation, int) else None
    
    def _get_weighted_candidates(self, heard, n=50):
        """Возвращает кандидатов с весами на основе популярности"""
        candidates = []
        weights = []
        
        for track in self.all_tracks:
            if track in heard:
                continue
            # Популярные треки имеют больший вес
            weight = self.popularity.get(track, 1)
            candidates.append(track)
            weights.append(weight)
        
        if not candidates:
            return []
        
        # Выбираем с учётом весов
        chosen = random.choices(candidates, weights=weights, k=min(n, len(candidates)))
        return chosen
    
    def recommend(self, observation, reward, done):
        if done:
            return None
        
        track_id = self._get_track_id(observation)
        if track_id is not None:
            self.history.append(track_id)
            self.popularity[track_id] += 1
        
        # Адаптивное окно: чем длиннее сессия, тем больше исключаем
        window_size = min(5, len(self.history) // 5 + 3)
        heard = set(self.history[-window_size:])
        
        # В начале сессии исключаем меньше, в конце — больше
        if len(self.history) < 5:
            heard = set(self.history[-2:])
        
        candidates = self._get_weighted_candidates(heard, n=30)
        
        if candidates:
            return random.choice(candidates)
        
        return self.action_space.sample()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass