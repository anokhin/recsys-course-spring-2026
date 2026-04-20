from typing import List, Dict, Any

from botify.recommenders import Recommender
from botify.recommenders.catboost_recommender import CatBoostRecommender
from botify.recommenders.sasrec_i2i import SasRecI2IRecommender
from botify.tracks import TracksCatalog
from botify.artists import ArtistsCatalog
from botify.users import UsersCatalog
from botify.experiment import Experiments, Treatment


class ABCatBoostRecommender(Recommender):
    """
    A/B тест: SasRec-I2I (Контроль) vs CatBoost (Тритмент).
    Использует хеш от user_id для стабильного разделения.
    """

    def __init__(
        self,
        tracks_catalog: TracksCatalog,
        artists_catalog: ArtistsCatalog,
        users_catalog: UsersCatalog,
    ):
        # Инициализируем оба рекомендера
        self.control = SasRecI2IRecommender(tracks_catalog, artists_catalog, users_catalog)
        self.treatment = CatBoostRecommender(tracks_catalog, artists_catalog, users_catalog)
        
        self.experiment = Experiments.CATBOOST_AB
        
        # Статистика для отладки
        self.control_count = 0
        self.treatment_count = 0

    def recommend(self, user_id: int, context: Dict[str, Any], n: int) -> List[int]:
        """
        Определяем группу по user_id и вызываем соответствующий рекомендер.
        """
        treatment = self.experiment.assign(user_id)
        
        if treatment == Treatment.C:
            # Контрольная группа: SasRec-I2I
            self.control_count += 1
            return self.control.recommend(user_id, context, n)
        else:
            # Тритмент группа: CatBoost
            self.treatment_count += 1
            return self.treatment.recommend(user_id, context, n)
    
    def get_stats(self) -> Dict[str, int]:
        """Возвращает статистику по распределению."""
        return {
            "control": self.control_count,
            "treatment": self.treatment_count,
        }