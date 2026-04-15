import mmh3
from typing import Dict, Any

from sim.agents.base import Recommender
from sim.agents.sasrec_i2i import SasRecI2IRecommender
from sim.agents.catboost_agent import CatBoostRecommender


class ABCatBoostRecommender(Recommender):
    """A/B тест: SasRec-I2I vs CatBoost."""

    def __init__(self, action_space):
        self.action_space = action_space
        self.control = SasRecI2IRecommender(action_space)
        self.treatment = CatBoostRecommender(action_space)
        
        self.experiment_hash = mmh3.hash("CATBOOST_REG_AB")
        self.user_assignment: Dict[int, str] = {}
        
        self.control_count = 0
        self.treatment_count = 0

    def recommend(self, observation: Dict[str, Any], reward: float, done: bool) -> int:
        user_id = observation.get("user", 0)
        
        if user_id not in self.user_assignment:
            user_hash = mmh3.hash(str(user_id), self.experiment_hash, False)
            if user_hash % 2 == 0:
                self.user_assignment[user_id] = "control"
                self.control_count += 1
            else:
                self.user_assignment[user_id] = "treatment"
                self.treatment_count += 1
        
        if self.user_assignment[user_id] == "control":
            return self.control.recommend(observation, reward, done)
        else:
            return self.treatment.recommend(observation, reward, done)

    def __enter__(self):
        self.control.__enter__()
        self.treatment.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.control.__exit__(exc_type, exc_val, exc_tb)
        self.treatment.__exit__(exc_type, exc_val, exc_tb)
        
        total = self.control_count + self.treatment_count
        if total > 0:
            print(f"\n[A/B Статистика]")
            print(f"  Control (SasRec-I2I): {self.control_count} ({100*self.control_count/total:.1f}%)")
            print(f"  Treatment (CatBoost): {self.treatment_count} ({100*self.treatment_count/total:.1f}%)")