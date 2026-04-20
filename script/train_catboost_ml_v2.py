import json
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


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
    
    emb_path = project_root / "sim" / "data" / "embeddings.npy"
    embeddings = np.load(emb_path)
    
    all_rewards = []
    for session in sessions:
        for event in session.get("events", []):
            all_rewards.append(event.get("listen_percent", 0))
    
    median_reward = np.median(all_rewards)
    print(f"Медианный listen_percent: {median_reward:.4f}")
    
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
            
            # Новые признаки
            unique_tracks = len(set(history))
            diversity = unique_tracks / len(history) if history else 0
            
            # Среднее сходство в истории
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
                diversity,           # Новый признак
                avg_hist_sim,        # Новый признак
            ]
            
            X.append(features)
            y.append(1 if next_reward > median_reward else 0)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    print(f"Размер данных: {X.shape}")
    print(f"Позитивных примеров: {y.sum()} ({100*y.mean():.1f}%)")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nОбучаем CatBoost...")
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        loss_function='Logloss',
        verbose=50,
        random_seed=42,
        early_stopping_rounds=30,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=50,
        plot=False,
    )
    
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"\nValidation AUC: {auc:.4f}")
    
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    model.save_model(str(models_dir / "cb_classifier.cbm"))
    
    meta = {
        "median_reward": float(median_reward),
        "feature_names": ["similarity", "history_len", "last_reward", "position", 
                         "user_norm", "track_norm", "diversity", "avg_hist_sim"],
        "auc": auc,
    }
    with open(models_dir / "cb_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Модель сохранена! AUC: {auc:.4f}")


if __name__ == "__main__":
    main()