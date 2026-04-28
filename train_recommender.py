import json, os
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from implicit.als import AlternatingLeastSquares

os.environ['OPENBLAS_NUM_THREADS'] = '1'

LOG_FILES = [
    "collected_data/tmp/data.json",
    "collected_data/tmp/data.json.1",
    "collected_data/tmp/data.json.2",
    "collected_data/tmp/data.json.3",
]

print("Читаем логи...")
user_events = defaultdict(list)
for log_file in LOG_FILES:
    try:
        with open(log_file) as f:
            for line in f:
                try: row = json.loads(line)
                except: continue
                user_events[row['user']].append(row)
    except FileNotFoundError:
        print(f"  Не найден: {log_file}")

print(f"Пользователей: {len(user_events)}")
pair_weights = defaultdict(lambda: defaultdict(float))
all_tracks = set()

for user, events in user_events.items():
    events.sort(key=lambda x: x['timestamp'])
    session = []
    for event in events:
        if event['message'] == 'next':
            t = float(event.get('time', 0))
            track = event['track']
            all_tracks.add(track)
            if session and t > 0.1:
                prev_track = session[-1][0]
                pair_weights[prev_track][track] += t
            session.append((track, t))
        elif event['message'] == 'last':
            session = []

all_tracks = sorted(all_tracks)
track2idx = {t: i for i, t in enumerate(all_tracks)}
print(f"Треков: {len(all_tracks)}")
print(f"Уникальных пар: {sum(len(v) for v in pair_weights.values())}")

rows, cols, data = [], [], []
for prev_track, next_tracks in pair_weights.items():
    pi = track2idx[prev_track]
    for next_track, weight in next_tracks.items():
        ni = track2idx[next_track]
        rows.append(pi)
        cols.append(ni)
        data.append(weight)

mat = sp.csr_matrix(
    (data, (rows, cols)),
    shape=(len(all_tracks), len(all_tracks)),
    dtype=np.float32
)
print(f"Матрица переходов: {mat.shape}, ненулевых: {mat.nnz}")

print("Обучаем ALS...")
model = AlternatingLeastSquares(
    factors=256,
    regularization=0.01,
    iterations=30,
    random_state=42,
)
model.fit(mat, show_progress=True)

prev_emb = model.user_factors   # (n_tracks, 256)
next_emb = model.item_factors   # (n_tracks, 256)
print(f"prev_emb: {prev_emb.shape}, next_emb: {next_emb.shape}")

print("Генерируем I2I рекомендации...")
prev_norms = np.linalg.norm(prev_emb, axis=1, keepdims=True)
prev_norms = np.where(prev_norms == 0, 1e-8, prev_norms)
next_norms = np.linalg.norm(next_emb, axis=1, keepdims=True)
next_norms = np.where(next_norms == 0, 1e-8, next_norms)

prev_norm = prev_emb / prev_norms
next_norm = next_emb / next_norms

TOP_K = 10
BATCH = 500
output_path = "botify/data/als_i2i_recommendations.jsonl"

with open(output_path, "w") as out:
    for start in range(0, len(all_tracks), BATCH):
        end = min(start + BATCH, len(all_tracks))
        scores = prev_norm[start:end] @ next_norm.T
        for i, score_row in enumerate(scores):
            track_idx = start + i
            track_id = all_tracks[track_idx]
            top_idx = np.argpartition(score_row, -(TOP_K+1))[-(TOP_K+1):]
            top_idx = top_idx[np.argsort(score_row[top_idx])[::-1]]
            recs = [all_tracks[idx] for idx in top_idx if idx != track_idx][:TOP_K]
            out.write(json.dumps({"item_id": track_id, "recommendations": recs}) + "\n")
        if start % 5000 == 0:
            print(f"  {end}/{len(all_tracks)}")

print(f"Готово! {output_path}")
with open(output_path) as f:
    for i, line in enumerate(f):
        if i >= 3: break
        print(" ", line.strip())