#!/usr/bin/env python3
"""
Обучение ALS (Alternating Least Squares) на собранных сессиях.
"""
import json
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
import pickle

# Пробуем импортировать implicit
try:
    from implicit.als import AlternatingLeastSquares
    IMPLICIT_AVAILABLE = True
except ImportError:
    print("Ошибка: implicit не установлен. Выполни: pip install implicit")
    IMPLICIT_AVAILABLE = False
    exit(1)


def main():
    project_root = Path(__file__).parent.parent
    
    # Загружаем все сессии
    sessions_file = project_root / "data" / "sasrec_sessions.jsonl"
    
    print("Загружаем сессии...")
    sessions = []
    with open(sessions_file) as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))
    
    print(f"Загружено сессий: {len(sessions)}")
    
    # Строим user-item матрицу
    user_ids = {}
    item_ids = {}
    rows = []
    cols = []
    data = []
    
    for session in sessions:
        user = session.get("user", 0)
        if user not in user_ids:
            user_ids[user] = len(user_ids)
        
        user_idx = user_ids[user]
        
        for event in session.get("events", []):
            item = event["track"]
            if item not in item_ids:
                item_ids[item] = len(item_ids)
            
            item_idx = item_ids[item]
            weight = event.get("listen_percent", 0.5)
            
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(weight)
    
    n_users = len(user_ids)
    n_items = len(item_ids)
    matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    
    print(f"Матрица: {n_users} пользователей, {n_items} треков")
    print(f"Ненулевых элементов: {matrix.nnz}")
    print(f"Плотность: {100 * matrix.nnz / (n_users * n_items):.4f}%")
    
    # Обучаем ALS
    print("\nОбучаем ALS...")
    model = AlternatingLeastSquares(
        factors=200,           # Размерность эмбеддингов
        regularization=0.01,
        iterations=50,
        random_state=42,
        use_gpu=False,
    )
    
    model.fit(matrix)
    print("Обучение завершено!")
    
    # Сохраняем модель
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    with open(models_dir / "als_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(models_dir / "als_mappings.pkl", "wb") as f:
        pickle.dump({
            "user_ids": user_ids,
            "item_ids": item_ids,
        }, f)
    
    print(f"\nМодель сохранена в {models_dir}")
    print(f"Пользователей: {n_users}, треков: {n_items}")


if __name__ == "__main__":
    main()