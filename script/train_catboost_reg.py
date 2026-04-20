#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def main():
    project_root = Path(__file__).parent.parent
    
    sessions_file = project_root / "data" / "sasrec_sessions.jsonl"
    
    print("Загружаем сессии...")
    sessions = []
    with open(sessions_file) as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))
    
    print(f"Сессий: {len(sessions)}")
    
    emb_path = project_root / "sim" / "data" / "embeddings.npy"
    embeddings = np.load(emb_path)
    
    X, y = [], []
    
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
            user_emb = np.mean(hist_embs, axis=0) if hist_embs else np.zeros(embeddings.shape[1])
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
            
            hist_std = np.std(hist_embs, axis=0).mean() if len(hist_embs) > 1 else 0
            
            features = [
                similarity,
                len(history) / 50.0,
                reward,
                float(i) / 50.0,
                np.linalg.norm(user_emb),
                np.linalg.norm(next_emb),
                diversity,
                avg_hist_sim,
                hist_std,
            ]
            
            X.append(features)
            y.append(next_reward)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"Размер данных: {X.shape}")
    print(f"Средний target: {y.mean():.4f}, std: {y.std():.4f}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nОбучаем CatBoost...")
    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        verbose=50,
        random_seed=42,
        thread_count=1,  # Важно для стабильности
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=50,
        plot=False,
    )
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"\nValidation RMSE: {rmse:.4f}")
    
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    model.save_model(str(models_dir / "cb_regressor.cbm"))
    
    meta = {
        'feature_names': ['similarity', 'history_len', 'last_reward', 'position',
                         'user_norm', 'track_norm', 'diversity', 'avg_hist_sim', 'history_std'],
        'rmse': float(rmse),
    }
    with open(models_dir / "cb_reg_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Модель сохранена! RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()