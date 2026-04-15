#!/usr/bin/env python3
"""
Обучение LightGBM для предсказания listen_percent.
"""
import json
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle


def main():
    project_root = Path(__file__).parent.parent
    
    # Загружаем данные
    sessions_file = project_root / "data" / "sasrec_sessions.jsonl"
    
    print("Загружаем сессии...")
    sessions = []
    with open(sessions_file) as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))
    
    print(f"Сессий: {len(sessions)}")
    
    # Загружаем эмбеддинги
    emb_path = project_root / "sim" / "data" / "embeddings.npy"
    embeddings = np.load(emb_path)
    print(f"Эмбеддинги: {embeddings.shape}")
    
    # Готовим данные
    X = []
    y = []
    
    for session in sessions:
        events = session.get("events", [])
        if len(events) < 2:
            continue
        
        history = []
        for i in range(len(events) - 1):
            current_track = events[i]["track"]
            reward = events[i].get("listen_percent", 0)
            history.append(current_track)
            
            next_track = events[i + 1]["track"]
            next_reward = events[i + 1].get("listen_percent", 0)
            
            if current_track >= len(embeddings) or next_track >= len(embeddings):
                continue
            
            hist_embs = [embeddings[t] for t in history if t < len(embeddings)]
            if hist_embs:
                user_emb = np.mean(hist_embs, axis=0)
            else:
                user_emb = np.zeros(embeddings.shape[1])
            
            next_emb = embeddings[next_track]
            
            similarity = np.dot(user_emb, next_emb) / (
                np.linalg.norm(user_emb) * np.linalg.norm(next_emb) + 1e-9
            )
            
            unique_tracks = len(set(history))
            diversity = unique_tracks / len(history) if history else 0
            
            if len(hist_embs) >= 2:
                hist_sims = []
                for j in range(len(hist_embs)-1):
                    sim = np.dot(hist_embs[j], hist_embs[j+1]) / (
                        np.linalg.norm(hist_embs[j]) * np.linalg.norm(hist_embs[j+1]) + 1e-9
                    )
                    hist_sims.append(sim)
                avg_hist_sim = np.mean(hist_sims)
            else:
                avg_hist_sim = 0
            
            features = [
                similarity,
                len(history) / 50.0,
                reward,
                float(i) / 50.0,
                np.linalg.norm(user_emb),
                np.linalg.norm(next_emb),
                diversity,
                avg_hist_sim,
                np.std(hist_embs, axis=0).mean() if len(hist_embs) > 1 else 0,
            ]
            
            X.append(features)
            y.append(next_reward)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"Размер данных: {X.shape}")
    print(f"Средний target: {y.mean():.4f}, std: {y.std():.4f}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nОбучаем LightGBM...")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'seed': 42,
        'num_threads': 1,  # Важно для macOS
    }
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"\nValidation RMSE: {rmse:.4f}")
    
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    with open(models_dir / "lgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    meta = {
        'feature_names': ['similarity', 'history_len', 'last_reward', 'position',
                         'user_norm', 'track_norm', 'diversity', 'avg_hist_sim', 'history_std'],
        'rmse': float(rmse),
    }
    with open(models_dir / "lgb_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Модель сохранена! RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()