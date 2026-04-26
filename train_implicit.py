"""
Обучение Implicit ALS на собранных логах и генерация I2I рекомендаций.
"""
import json
import numpy as np
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

# 1. Загружаем логи
print("Загрузка логов...")
logs = []
with open("sim/data/logs/data.json", "r") as f:
    for line in f:
        logs.append(json.loads(line))

user_items = defaultdict(lambda: defaultdict(float))
for event in logs:
    if "recommendation" in event and event["recommendation"] is not None:
        user = event["user"]
        track = event["recommendation"]
        time = event.get("time", 0.5)
        user_items[user][track] += min(time, 1.0)

min_interactions = 2
active_users = {u for u, items in user_items.items() if len(items) >= min_interactions}
print(f"Активных пользователей: {len(active_users)}")

# 2. Строим матрицу
user_list = list(active_users)
item_set = set()
for u in user_list:
    item_set.update(user_items[u].keys())
item_list = list(item_set)

user_idx = {u: i for i, u in enumerate(user_list)}
item_idx = {t: i for i, t in enumerate(item_list)}

row, col, data = [], [], []
for user, items in user_items.items():
    if user in user_idx:
        for item, weight in items.items():
            if item in item_idx:
                row.append(item_idx[item])
                col.append(user_idx[user])
                data.append(weight)

item_user_matrix = sparse.coo_matrix((data, (row, col)), shape=(len(item_list), len(user_list))).tocsr()
print(f"Матрица: {item_user_matrix.shape[0]} треков x {item_user_matrix.shape[1]} пользователей")

# 3. Обучаем ALS
print("Обучение ALS...")
model = AlternatingLeastSquares(factors=64, regularization=0.1, iterations=30)
model.fit(item_user_matrix)

# Эмбеддинги треков - это user_factors (т.к. мы транспонировали матрицу)
track_embeddings = model.user_factors
print(f"Track embeddings: {track_embeddings.shape}")

# 4. Вычисляем I2I батчами
print("Вычисление похожести треков...")
num_items = len(item_list)
i2i_recommendations = {}

batch_size = 5000
for start in range(0, num_items, batch_size):
    end = min(start + batch_size, num_items)
    batch_embs = track_embeddings[start:end]
    
    # Косинусная близость батча ко всем трекам
    sim_batch = cosine_similarity(batch_embs, track_embeddings)
    
    for i in range(sim_batch.shape[0]):
        global_idx = start + i
        tid = item_list[global_idx]
        
        # Исключаем сам трек
        sim_batch[i, global_idx] = -1
        
        # Топ-10 похожих
        top_indices = np.argsort(sim_batch[i])[-10:][::-1]
        top_tracks = [int(item_list[j]) for j in top_indices]
        i2i_recommendations[tid] = top_tracks
    
    print(f"  Обработано {end}/{num_items} треков")

# 5. Сохраняем
print("Сохранение результатов...")
with open("botify/data/implicit_ml_i2i.jsonl", "w") as f:
    for tid, recs in i2i_recommendations.items():
        f.write(json.dumps({"item_id": int(tid), "recommendations": recs}) + "\n")

print(f"✅ Сохранено {len(i2i_recommendations)} I2I рекомендаций в botify/data/implicit_ml_i2i.jsonl")
