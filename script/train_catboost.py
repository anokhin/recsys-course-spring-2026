import json
import numpy as np
from pathlib import Path
import catboost


def main():
    project_root = Path(__file__).parent.parent
    
    # Загружаем эмбеддинги
    embeddings_path = project_root / "sim" / "data" / "embeddings.npy"
    track_embeddings = np.load(embeddings_path)
    print(f"Эмбеддинги: {track_embeddings.shape}")
    
    # Простая модель
    print("Создаю CatBoost модель...")
    model = catboost.CatBoostRanker(
        iterations=100,
        learning_rate=0.1,
        depth=4,
        verbose=False,
    )
    
    # Dummy данные для начала
    X = np.random.randn(1000, 4)
    y = np.random.rand(1000) * 100
    groups = np.repeat(np.arange(100), 10)
    
    model.fit(X, y, group_id=groups, verbose=False)
    
    # Сохраняем
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "cb_ranker.cbm"
    model.save_model(str(model_path))
    print(f"Модель сохранена: {model_path}")
    
    print("\nВАЖНО: Это базовая модель. Для улучшения:")
    print("1. Собери данные: python -m sim.run single --recommender sasrec --episodes 50000")
    print("2. Переобучи модель с реальными данными")


if __name__ == "__main__":
    main()