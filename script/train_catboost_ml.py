import json
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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
    
    # Загружаем эмбеддинги (разрешено, это features, а не данные пользователей)
    emb_path = project_root / "sim" / "data" / "embeddings.npy"
    embeddings = np.load(emb_path)
    
    # Собираем все listen_percent для определения медианы
    all_rewards = []
    for session in sessions:
        for event in session.get("events", []):
            all_rewards.append(event.get("listen_percent", 0))
    
    median_reward = np.median(all_rewards)
    print(f"Медианный listen_percent: {median_reward:.4f}")
    
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
            
            # Эмбеддинг пользователя = среднее по истории
            hist_embs = [embeddings[t] for t in history if t < len(embeddings)]
            if hist_embs:
                user_emb = np.mean(hist_embs, axis=0)
            else:
                user_emb = np.zeros(embeddings.shape[1])
            
            # Эмбеддинг следующего трека
            next_emb = embeddings[next_track]
            
            # Фичи
            similarity = np.dot(user_emb, next_emb) / (
                np.linalg.norm(user_emb) * np.linalg.norm(next_emb) + 1e-9
            )
            
            features = [
                similarity,
                len(history) / 50.0,
                reward,                        # reward текущего трека
                float(i) / 50.0,
                np.linalg.norm(user_emb),
                np.linalg.norm(next_emb),
            ]
            
            X.append(features)
            # Бинарный таргет: 1 если следующий трек прослушан лучше медианы
            y.append(1 if next_reward > median_reward else 0)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    print(f"Размер данных: {X.shape}")
    print(f"Позитивных примеров: {y.sum()} ({100*y.mean():.1f}%)")
    
    # Разделяем на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Обучаем CatBoost
    print("\nОбучаем CatBoost...")
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        verbose=50,
        random_seed=42,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=50,
        plot=False,
    )
    
    # Оцениваем
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"\nValidation AUC: {auc:.4f}")
    
    # Сохраняем модель и метаданные
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    model.save_model(str(models_dir / "cb_classifier.cbm"))
    
    meta = {
        "median_reward": float(median_reward),
        "feature_names": ["similarity", "history_len", "last_reward", "position", "user_norm", "track_norm"],
    }
    with open(models_dir / "cb_meta.json", "w") as f:
        json.dump(meta, f)
    
    print("Модель сохранена!")


if __name__ == "__main__":
    main()