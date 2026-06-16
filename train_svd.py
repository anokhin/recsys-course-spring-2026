"""
Обучение SVD на логах и генерация I2I рекомендаций.
"""
import json
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
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

# Строим user-item матрицу (пользователи x треки)
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

user_list = list(active_users)
item_set = set()
for u in user_list:
    item_set.update(user_items[u].keys())
item_list = list(item_set)

user_idx = {u: i for i, u in enumerate(user_list)}
item_idx = {t: i for i, t in enumerate(item_list)}

# Строим user-item матрицу (пользователи как строки)
row, col, data = [], [], []
for user, items in user_items.items():
    if user in user_idx:
        for item, weight in items.items():
            if item in item_idx:
                row.append(user_idx[user])
                col.append(item_idx[item])
                data.append(weight)

user_item_matrix = sparse.coo_matrix((data, (row, col)), shape=(len(user_list), len(item_list))).tocsr()
print(f"Матрица: {user_item_matrix.shape[0]} пользователей x {user_item_matrix.shape[1]} треков")

# 2. Обучаем SVD на item embeddings
print("Обучение SVD...")
svd = TruncatedSVD(n_components=100, random_state=42, n_iter=15)
item_embeddings = svd.fit_transform(user_item_matrix.T)  # транспонируем: items x users -> items x components
print(f"Item embeddings: {item_embeddings.shape}")
print(f"Explained variance: {svd.explained_variance_ratio_.sum():.4f}")

# 3. Вычисляем I2I
print("Вычисление похожести треков...")
num_items = len(item_list)
i2i_recommendations = {}

batch_size = 5000
for start in range(0, num_items, batch_size):
    end = min(start + batch_size, num_items)
    batch_embs = item_embeddings[start:end]
    
    sim_batch = cosine_similarity(batch_embs, item_embeddings)
    
    for i in range(sim_batch.shape[0]):
        global_idx = start + i
        tid = item_list[global_idx]
        
        sim_batch[i, global_idx] = -1
        top_indices = np.argsort(sim_batch[i])[-10:][::-1]
        top_tracks = [int(item_list[j]) for j in top_indices]
        i2i_recommendations[tid] = top_tracks
    
    print(f"  Обработано {end}/{num_items} треков")

# 4. Сохраняем
print("Сохранение результатов...")
with open("botify/data/svd_ml_i2i.jsonl", "w") as f:
    for tid, recs in i2i_recommendations.items():
        f.write(json.dumps({"item_id": int(tid), "recommendations": recs}) + "\n")

print(f"✅ Сохранено {len(i2i_recommendations)} I2I рекомендаций в botify/data/svd_ml_i2i.jsonl")
