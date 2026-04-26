"""
Предобучение ContextualBanditRecommender на логах симулятора.
"""
import json
import pickle
import numpy as np
import sys
sys.path.insert(0, "botify")

from botify.recommenders.contextual_bandit import ContextualBanditRecommender
from botify.track import Catalog

# Загружаем логи
logs = []
with open("sim/data/logs/data.json", "r") as f:
    for line in f:
        logs.append(json.loads(line))

print(f"Загружено {len(logs)} событий")

# Загружаем треки
class FakeApp:
    def __init__(self):
        self.config = {"TRACKS_CATALOG": "botify/data/tracks.json"}
        self.logger = type('obj', (object,), {'info': print})

catalog = Catalog(FakeApp()).load("botify/data/tracks.json")
print(f"Загружено {len(catalog.tracks)} треков")

# Обучаем bandit на истории пользователей
# Группируем логи по пользователям
user_sessions = {}
for event in logs:
    user = event["user"]
    if user not in user_sessions:
        user_sessions[user] = []
    user_sessions[user].append(event)

print(f"Пользователей: {len(user_sessions)}")

# Обучаем (имитируем работу recommend_next с reward по времени прослушивания)
# Это offline обучение: прогоняем все события через модель
class FakeRedis:
    def __init__(self):
        self.data = {}
    def lrange(self, key, start, end):
        return self.data.get(key, [])
    def get(self, key):
        return None

fake_listen_redis = FakeRedis()
fake_track_redis = FakeRedis()
fake_fallback = type('obj', (object,), {'recommend_next': lambda u, t, tm: 0})

bandit = ContextualBanditRecommender(
    fake_listen_redis,
    fake_track_redis,
    catalog,
    fake_fallback,
    alpha=1.5
)

# Прогоняем историю через модель для обучения
trained_users = 0
for user, events in user_sessions.items():
    # Сортируем по времени
    events.sort(key=lambda e: e["timestamp"])
    
    for event in events:
        if "track" in event and "time" in event:
            # Обновляем модель на основе прослушивания
            bandit._update_model(user, event["track"], event["time"])
    
    trained_users += 1
    if trained_users % 1000 == 0:
        print(f"  Обучено {trained_users} пользователей")

print(f"✅ Обучено {trained_users} пользователей")

# Сохраняем обученные параметры
pretrained_data = {
    "A": {str(k): v.tolist() for k, v in bandit.A.items()},
    "b": {str(k): v.tolist() for k, v in bandit.b.items()},
    "feature_dim": bandit.d
}

with open("botify/data/pretrained_bandit.pkl", "wb") as f:
    pickle.dump(pretrained_data, f)

print(f"✅ Параметры сохранены в botify/data/pretrained_bandit.pkl")
print(f"  Пользователей: {len(pretrained_data['A'])}")
print(f"  Размерность фич: {pretrained_data['feature_dim']}")
