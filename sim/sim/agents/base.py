from typing import Dict, Any


class Recommender:
    """Базовый класс для всех рекомендеров."""

    def recommend(self, observation: Dict[str, Any], reward: float, done: bool) -> int:
        """Возвращает ID рекомендованного трека."""
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass