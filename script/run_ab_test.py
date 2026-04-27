import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.run import run_episode
from sim.envs import RecEnv
from sim.envs.config import RecEnvConfigSchema
from sim.agents.remote import RemoteRecommender
import yaml
import tqdm
import numpy as np
from scipy import stats


def run_ab_test(episodes: int):
    config = RecEnvConfigSchema().load(yaml.full_load(open("sim/config/env.yml")))
    
    control_rewards = []
    treatment_rewards = []
    control_steps = []
    treatment_steps = []
    
    with RecEnv(config) as env:
        env.seed(42)
        recommender = RemoteRecommender(config.remote_recommender_config)
        
        with recommender, tqdm.tqdm(total=episodes) as progress:
            for episode_id in range(episodes):
                observation, _ = env.reset()
                done = False
                reward_sum = 0.0
                steps = 0
                user_id = observation.get("user", 0)
                
                while not done:
                    action = recommender.recommend(observation, reward_sum, done)
                    observation, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    reward_sum += reward
                    steps += 1
                
                recommender.recommend(observation, reward_sum, done)
                
                # Определяем группу по user_id (как в Experiments.HSTU)
                # HSTU использует хеш user_id % 2
                if user_id % 2 == 0:
                    control_rewards.append(reward_sum)
                    control_steps.append(steps)
                else:
                    treatment_rewards.append(reward_sum)
                    treatment_steps.append(steps)
                
                progress.update(1)
                progress.set_postfix({
                    'C': f'{np.mean(control_rewards):.2f}' if control_rewards else '-',
                    'T': f'{np.mean(treatment_rewards):.2f}' if treatment_rewards else '-'
                })
    
    # Вывод результатов
    print("\n" + "=" * 50)
    print("A/B Test Results")
    print("=" * 50)
    
    print(f"\nКонтроль (SasRec-I2I):")
    print(f"  Сессий: {len(control_rewards)}")
    print(f"  Средний reward: {np.mean(control_rewards):.4f} ± {np.std(control_rewards)/np.sqrt(len(control_rewards)):.4f}")
    print(f"  Средние шаги: {np.mean(control_steps):.4f}")
    
    print(f"\nТритмент (CatBoost):")
    print(f"  Сессий: {len(treatment_rewards)}")
    print(f"  Средний reward: {np.mean(treatment_rewards):.4f} ± {np.std(treatment_rewards)/np.sqrt(len(treatment_rewards)):.4f}")
    print(f"  Средние шаги: {np.mean(treatment_steps):.4f}")
    
    # Статистический тест
    if len(control_rewards) > 0 and len(treatment_rewards) > 0:
        t_stat, p_value = stats.ttest_ind(treatment_rewards, control_rewards)
        improvement = (np.mean(treatment_rewards) / np.mean(control_rewards) - 1) * 100
        
        print(f"\nРезультат:")
        print(f"  Прирост: {improvement:+.2f}%")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Значимо: {'ДА (p < 0.05)' if p_value < 0.05 else 'НЕТ'}")
    
    return {
        'control_rewards': control_rewards,
        'treatment_rewards': treatment_rewards,
        'control_steps': control_steps,
        'treatment_steps': treatment_steps,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    args = parser.parse_args()
    
    run_ab_test(args.episodes)