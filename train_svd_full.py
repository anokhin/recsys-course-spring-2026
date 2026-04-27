"""
Обучение SVD на ВСЕХ собранных логах (старые + новые 90K).
"""
import json
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import glob
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

# 1. Загружаем ВСЕ логи
print("Загрузка ВСЕХ логов...")
all_logs = []

# Старые логи
for log_file in glob.glob("sim/data/logs*/data.json"):
    print(f"  Читаем {log_file}...")
    with open(log_file, "r") as f:
        for line in f:
            all_logs.append(json.loads(line))

print(f"Всего событий: {len(all_logs)}")

# Строим user-item матрицу с TF-IDF взвешиванием
user_items = defaultdict(lambda: defaultdict(float))
track_users = defaultdict(set)

for event in all_logs:
    if "recommendation" in event and event["recommendation"] is not None:
        user = event["user"]
        track = event["recommendation"]
        time = event.get("time", 0.5)
        weight = min(time, 1.0)
        user_items[user][track] += weight
        track_users[track].add(user)

min_interactions = 3  # повысили порог
active_users = {u for u, items in user_items.items() if len(items) >= min_interactions}
print(f"Активных пользователей: {len(active_users)}")

user_list = list(active_users)
item_set = set()
for u in user_list:
    item_set.update(user_items[u].keys())
item_list = list(item_set)

print(f"Треков: {len(item_list)}")

user_idx = {u: i for i, u in enumerate(user_list)}
item_idx = {t: i for i, t in enumerate(item_list)}

# TF-IDF взвешивание
n_users = len(user_list)
row, col, data = [], [], []
for user, items in user_items.items():
    if user in user_idx:
        for item, tf in items.items():
            if item in item_idx:
                idf = np.log(n_users / (1 + len(track_users[item])))
                weight = (1.0 + np.log1p(tf * 3)) * idf  # TF-IDF formula
                row.append(user_idx[user])
                col.append(item_idx[item])
                data.append(weight)

user_item_matrix = sparse.coo_matrix((data, (row, col)), shape=(n_users, len(item_list))).tocsr()
print(f"Матрица: {user_item_matrix.shape}")

# 2. Обучаем SVD
print("Обучение SVD с 200 компонентами...")
svd = TruncatedSVD(n_components=200, random_state=42, n_iter=20)
item_embeddings = svd.fit_transform(user_item_matrix.T)
print(f"Item embeddings: {item_embeddings.shape}")
print(f"Explained variance: {svd.explained_variance_ratio_.sum():.4f}")

# 3. I2I рекомендации
print("Вычисление I2I рекомендаций...")
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
with open("botify/data/svd_full_i2i.jsonl", "w") as f:
    for tid, recs in i2i_recommendations.items():
        f.write(json.dumps({"item_id": int(tid), "recommendations": recs}) + "\n")

print(f"✅ Сохранено {len(i2i_recommendations)} I2I рекомендаций в botify/data/svd_full_i2i.jsonl")
