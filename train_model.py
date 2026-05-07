import json
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
import lightgbm as lgb
import warnings
import gc

warnings.filterwarnings('ignore')

print("1. Загружаем логи...")
logs = []
with open('botify/log/data.json', 'r') as f:
    for line in f:
        try:
            logs.append(json.loads(line))
        except:
            continue

df = pd.DataFrame(logs)
print(f"Загружено {len(df)} событий")

tracks_meta = {}
with open('sim/data/tracks.json', 'r') as f:
    for line in f:
        try:
            track = json.loads(line)
            tracks_meta[track['track']] = track
        except:
            continue
print(f"Загружено треков: {len(tracks_meta)}")

print("\n2. Загружаем эмбеддинги...")
embeddings_orig = np.load('sim/data/embeddings.npy')
embeddings_norm = embeddings_orig / (np.linalg.norm(embeddings_orig, axis=1, keepdims=True) + 1e-8)
embeddings_norm = embeddings_norm.astype(np.float16)
max_track_id = len(embeddings_norm)
del embeddings_orig
gc.collect()
print(f"Эмбеддинги: {embeddings_norm.shape}")

print("\n3. Считаем популярность треков...")
track_stats = defaultdict(lambda: {'plays': 0, 'total_time': 0, 'skips': 0})
for _, row in df.iterrows():
    track = int(row['track'])
    time = float(row['time'])
    track_stats[track]['plays'] += 1
    track_stats[track]['total_time'] += time
    if time < 0.3:
        track_stats[track]['skips'] += 1

track_popularity = {}
for track, stats in track_stats.items():
    if stats['plays'] > 0:
        track_popularity[track] = {
            'play_count': stats['plays'],
            'avg_listen_time': stats['total_time'] / stats['plays'],
            'skip_rate': stats['skips'] / stats['plays']
        }
print(f"Статистика по {len(track_popularity)} трекам")

print("\n4. Собираем историю пользователей...")
user_history = defaultdict(list)
for _, row in df.iterrows():
    user = int(row['user'])
    user_history[user].append({
        'track': int(row['track']),
        'time': float(row['time']),
        'recommendation': int(row['recommendation']) if pd.notna(row.get('recommendation')) and row['recommendation'] is not None else None
    })

# Ограничиваем историю
for user in list(user_history.keys()):
    if len(user_history[user]) > 50:
        user_history[user] = user_history[user][-50:]

print(f"Пользователей: {len(user_history)}")

print("\n5. Создаём обучающие примеры...")
training_data = []

for user, history in user_history.items():
    if len(history) < 10:
        continue
    
    all_times = [h['time'] for h in history]
    all_tracks = [h['track'] for h in history if h['track'] < max_track_id]
    
    user_avg_time = np.mean(all_times)
    user_std_time = np.std(all_times) if len(all_times) > 1 else 0.1
    user_skip_rate = sum(1 for t in all_times if t < 0.3) / len(all_times)
    
    # User embedding
    if len(all_tracks) > 0:
        user_emb = np.zeros(embeddings_norm.shape[1], dtype=np.float32)
        for t in all_tracks[-500:]:
            user_emb += embeddings_norm[t].astype(np.float32)
        user_emb /= min(len(all_tracks), 500)
    else:
        user_emb = np.zeros(embeddings_norm.shape[1], dtype=np.float32)
    
    for i in range(10, len(history)):
        past = history[max(0, i-20):i]
        current = history[i]
        
        current_track = current['track']
        current_time = current['time']
        recommended_track = current['recommendation']
        
        if recommended_track is None:
            continue
        if current_track >= max_track_id or recommended_track >= max_track_id:
            continue
        
        past_tracks = [h['track'] for h in past if h['track'] < max_track_id]
        past_times = [h['time'] for h in past]
        
        if len(past_tracks) < 3:
            continue
        
        listen_time = current_time
        skip = 1 if current_time < 0.3 else 0
        n_past = len(past_tracks)
        
        # Эмбеддинги
        past_embs = embeddings_norm[past_tracks].astype(np.float32)
        track_emb = embeddings_norm[current_track].astype(np.float32)
        rec_emb = embeddings_norm[recommended_track].astype(np.float32)
        
        # 1. Простое среднее
        history_emb_avg = np.mean(past_embs, axis=0)
        
        # 2. Exponential decay
        time_weights = np.exp(-0.05 * np.arange(n_past)[::-1]).astype(np.float32)
        time_weights = time_weights / time_weights.sum()
        history_emb_exp = np.sum(past_embs * time_weights.reshape(-1, 1), axis=0)
        
        # 3. Listen-time weights
        listen_weights = np.array(past_times[-n_past:], dtype=np.float32)
        listen_weights = listen_weights / (listen_weights.sum() + 1e-8)
        history_emb_time = np.sum(past_embs * listen_weights.reshape(-1, 1), axis=0)
        
        # Сходства
        features = {
            'cos_avg_current': float(np.dot(history_emb_avg, track_emb)),
            'cos_avg_rec': float(np.dot(history_emb_avg, rec_emb)),
            'cos_exp_current': float(np.dot(history_emb_exp, track_emb)),
            'cos_exp_rec': float(np.dot(history_emb_exp, rec_emb)),
            'cos_time_current': float(np.dot(history_emb_time, track_emb)),
            'cos_time_rec': float(np.dot(history_emb_time, rec_emb)),
            'cos_current_rec': float(np.dot(track_emb, rec_emb)),
            'cos_user_current': float(np.dot(user_emb, track_emb)),
            'cos_user_rec': float(np.dot(user_emb, rec_emb)),
            'history_length': n_past,
            'session_position': i / len(history),
            'avg_history_time': np.mean(past_times[-10:]),
            'std_history_time': np.std(past_times[-10:]) if n_past >= 2 else 0,
            'skip_ratio_recent': sum(1 for t in past_times[-5:] if t < 0.3) / min(5, n_past),
            'skip_ratio_all': sum(1 for t in past_times if t < 0.3) / n_past,
            'user_avg_time': user_avg_time,
            'user_std_time': user_std_time,
            'user_skip_rate': user_skip_rate,
            'user_history_len': len(history),
            'unique_tracks_ratio': len(set(past_tracks[-10:])) / min(10, n_past),
        }
        
        # Популярность
        cp = track_popularity.get(current_track, {})
        features['current_play_count'] = np.log1p(cp.get('play_count', 0))
        features['current_avg_time'] = cp.get('avg_listen_time', 0.5)
        features['current_skip_rate'] = cp.get('skip_rate', 0.5)
        
        rp = track_popularity.get(recommended_track, {})
        features['rec_play_count'] = np.log1p(rp.get('play_count', 0))
        features['rec_avg_time'] = rp.get('avg_listen_time', 0.5)
        features['rec_skip_rate'] = rp.get('skip_rate', 0.5)
        
        # Метаданные
        rm = tracks_meta.get(recommended_track, {})
        features['rec_year'] = rm.get('year', 2000)
        features['rec_fans'] = rm.get('artist_fans', 50)
        
        training_data.append({**features, 'listen_time': listen_time, 'skip': skip})

df_train = pd.DataFrame(training_data)
print(f"Создано {len(df_train)} примеров, признаков: {len(df_train.columns) - 2}")
print(f"Среднее время: {df_train['listen_time'].mean():.3f}, Skip rate: {df_train['skip'].mean():.3f}")

# Чистим
df_train = df_train.fillna(0)
feature_cols = [c for c in df_train.columns if c not in ['listen_time', 'skip']]
for col in feature_cols:
    df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0)

X = df_train[feature_cols].astype(float)
y_reg = df_train['listen_time'].astype(float)
y_clf = df_train['skip'].astype(int)

constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    print(f"Удаляем: {constant_cols}")
    X = X.drop(constant_cols, axis=1)

print(f"\n6. Обучаем модели ({len(X)} примеров)...")
X_train, X_val, y_reg_train, y_reg_val, y_clf_train, y_clf_val = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

print("\n--- Регрессор ---")
model_reg = lgb.LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.03, num_leaves=63,
                               subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                               random_state=42, verbose=-1)
model_reg.fit(X_train, y_reg_train, eval_set=[(X_val, y_reg_val)], eval_metric='rmse',
              callbacks=[lgb.early_stopping(50, verbose=10)])
rmse = np.sqrt(mean_squared_error(y_reg_val, model_reg.predict(X_val)))
print(f"RMSE: {rmse:.4f}")

print("\n--- Классификатор ---")
model_clf = lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.03, num_leaves=63,
                                subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
                                random_state=42, verbose=-1)
model_clf.fit(X_train, y_clf_train, eval_set=[(X_val, y_clf_val)], eval_metric='auc',
              callbacks=[lgb.early_stopping(50, verbose=10)])
auc = roc_auc_score(y_clf_val, model_clf.predict_proba(X_val)[:, 1])
print(f"ROC-AUC: {auc:.4f}")

print(f"\nКорреляция ансамбля: {np.corrcoef(model_reg.predict(X_val) * (1 - model_clf.predict_proba(X_val)[:, 1]), y_reg_val)[0, 1]:.4f}")

print("\n7. Сохраняем...")
import os
os.makedirs('botify/botify', exist_ok=True)
with open('botify/botify/model_reg.pkl', 'wb') as f: pickle.dump(model_reg, f)
with open('botify/botify/model_clf.pkl', 'wb') as f: pickle.dump(model_clf, f)
np.save('botify/botify/embeddings.npy', embeddings_norm)
with open('botify/botify/feature_names.pkl', 'wb') as f: pickle.dump(list(X.columns), f)
with open('botify/botify/tracks_meta.pkl', 'wb') as f: pickle.dump(tracks_meta, f)
with open('botify/botify/track_popularity.pkl', 'wb') as f: pickle.dump(track_popularity, f)
print("\n✅ Готово!")