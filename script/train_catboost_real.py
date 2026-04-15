import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import catboost


def load_sessions(file_path: Path):
    """Загружает сессии из JSON Lines файла."""
    sessions = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))
    return sessions


def prepare_training_data(sessions, track_embeddings):
    """Готовит данные для обучения ранжированию."""
    features = []
    targets = []
    groups = []
    
    group_id = 0
    
    for session in sessions:
        events = session.get("events", [])
        if len(events) < 2:
            continue
        
        history_tracks = []
        history_rewards = []
        
        for i, event in enumerate(events[:-1]):
            track = event["track"]
            reward = event["listen_percent"]
            
            history_tracks.append(track)
            history_rewards.append(reward)
            
            # Эмбеддинг пользователя
            valid_embs = [track_embeddings[t] for t in history_tracks if t < len(track_embeddings)]
            if valid_embs:
                user_emb = np.mean(valid_embs, axis=0)
            else:
                user_emb = np.zeros(track_embeddings.shape[1])
            
            # Следующий трек
            next_event = events[i + 1]
            next_track = next_event["track"]
            target = next_event["listen_percent"]
            
            if next_track >= len(track_embeddings):
                continue
            
            track_emb = track_embeddings[next_track]
            
            # Косинусное сходство
            similarity = np.dot(user_emb, track_emb) / (
                np.linalg.norm(user_emb) * np.linalg.norm(track_emb) + 1e-9
            )
            
            # Средний reward в сессии
            avg_reward = np.mean(history_rewards) if history_rewards else 0.0
            
            # Фичи (масштабирование подобрано под нормализованные rewards)
            feat = [
                similarity,                    # Косинусное сходство (обычно 0.1-0.4)
                len(history_tracks) / 50.0,    # Длина истории (нормировка)
                avg_reward,                    # Средний reward (уже в 0-1)
                float(i) / 50.0,               # Позиция в сессии
            ]
            
            features.append(feat)
            targets.append(target)
            groups.append(group_id)
        
        group_id += 1
    
    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32), np.array(groups)


def main():
    project_root = Path(__file__).parent.parent
    
    # Загружаем данные
    sessions_file = project_root / "data" / "sasrec_sessions.jsonl"
    if not sessions_file.exists():
        print(f"Файл {sessions_file} не найден!")
        print("Сначала запусти: cd sim && python3 sim/data_collector.py --episodes 10000")
        return
    
    print("Загружаем сессии...")
    sessions = load_sessions(sessions_file)
    print(f"Загружено {len(sessions)} сессий")
    
    # Загружаем эмбеддинги
    embeddings_path = project_root / "sim" / "data" / "embeddings.npy"
    track_embeddings = np.load(embeddings_path)
    print(f"Эмбеддинги: {track_embeddings.shape}")
    
    # Готовим данные
    print("Готовим обучающие данные...")
    X, y, groups = prepare_training_data(sessions, track_embeddings)
    print(f"Обучающая выборка: {X.shape}")
    print(f"Средний target: {y.mean():.2f}")
    
    # Разделяем на train/val
    val_size = min(2000, len(X) // 5)
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]
    groups_train = groups[:-val_size]
    groups_val = groups[-val_size:]
    
    # Обучаем CatBoost
    print("\nОбучаем CatBoost...")
    model = catboost.CatBoostRanker(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="YetiRank",
        verbose=50,
        random_seed=42,
    )
    
    # Создаём Pool объекты для валидации
    train_pool = catboost.Pool(X_train, y_train, group_id=groups_train)
    val_pool = catboost.Pool(X_val, y_val, group_id=groups_val)
    
    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=50,
        plot=False,
    )
    
    # Сохраняем модель
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "cb_ranker.cbm"
    model.save_model(str(model_path))
    print(f"\nМодель сохранена в {model_path}")
    
    # Проверяем качество на валидации
    preds = model.predict(X_val)
    print(f"\nNDCG на валидации: {model.score(X_val, y_val):.4f}")


if __name__ == "__main__":
    main()