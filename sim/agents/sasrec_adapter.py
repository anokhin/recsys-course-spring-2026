import sys
import os
import random
import pickle

# Добавляем путь к botify
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'botify'))

class SasrecAdapter:
    def __init__(self, action_space):
        self.action_space = action_space
        self.name = "SasRecI2I"
        self.recommender = None
        self.history = []
        self.all_tracks = list(range(2000))
        
    def _get_recommender(self):
        if self.recommender is None:
            try:
                from botify.recommenders.i2i import I2IRecommender
                
                # Создаём мок-объекты для Redis (для симулятора)
                class MockRedis:
                    def __init__(self):
                        self.data = {}
                    
                    def get(self, key):
                        return self.data.get(key)
                    
                    def set(self, key, value):
                        self.data[key] = value
                    
                    def lrange(self, key, start, end):
                        return []
                    
                    def lpush(self, key, value):
                        pass
                    
                    def ltrim(self, key, start, end):
                        pass
                
                mock_redis = MockRedis()
                
                # Загружаем рекомендации из файла, если есть
                sasrec_path = os.path.join(os.path.dirname(__file__), '..', '..', 'botify', 'data', 'sasrec_i2i.jsonl')
                if os.path.exists(sasrec_path):
                    import json
                    with open(sasrec_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                mock_redis.set(data['item_id'], pickle.dumps(data['recommendations']))
                    print(f"✅ Загружены рекомендации SasRec из {sasrec_path}")
                
                self.recommender = I2IRecommender(mock_redis, mock_redis, None)
                print("✅ SasRec-I2I адаптер создан")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки SasRec: {e}")
                self.recommender = None
        return self.recommender
    
    def _get_track_id(self, observation):
        if observation is None:
            return None
        if isinstance(observation, dict):
            return observation.get('track_id', observation.get('id', None))
        return observation if isinstance(observation, int) else None
    
    def recommend(self, observation, reward, done):
        if done:
            return None
        
        track_id = self._get_track_id(observation)
        if track_id is not None:
            self.history.append(track_id)
        
        rec = self._get_recommender()
        if rec is not None and track_id is not None:
            try:
                result = rec.recommend_next(0, track_id, reward)
                return int(result)
            except Exception as e:
                print(f"Ошибка recommend_next: {e}")
        
        # Fallback: случайный трек, не повторяя последние 3
        heard = set(self.history[-3:])
        candidates = [t for t in self.all_tracks if t not in heard]
        if candidates:
            return random.choice(candidates)
        
        return self.action_space.sample()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass