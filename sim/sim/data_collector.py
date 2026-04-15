import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.run import run_episode, TimeControl
from sim.envs import RecEnv
from sim.envs.config import RecEnvConfigSchema, RecEnvConfig
from sim.agents import SasRecI2IRecommender
import yaml
import tqdm


def collect_training_data(episodes: int, output_file: str):
    """Собирает сырые данные сессий для обучения."""
    config = RecEnvConfigSchema().load(yaml.full_load(open("config/env.yml")))
    
    sessions = []
    
    with RecEnv(config) as env:
        env.seed(42)
        recommender = SasRecI2IRecommender(env.action_space)
        
        with recommender, tqdm.tqdm(total=episodes) as progress:
            for episode_id in range(episodes):
                # Запускаем эпизод и собираем данные
                observation, _ = env.reset()
                done = False
                reward = 1.0
                
                session_data = {
                    "episode": episode_id,
                    "user": observation.get("user", 0),
                    "events": []
                }
                
                while not done:
                    action = recommender.recommend(observation, reward, done)
                    
                    # Сохраняем событие ДО шага
                    event = {
                        "track": int(action),
                        "user": observation.get("user", 0),
                        "time": observation.get("time", 0),
                    }
                    
                    observation, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    # Добавляем результат после шага
                    event["listen_percent"] = float(reward)
                    session_data["events"].append(event)
                
                recommender.recommend(observation, reward, done)
                sessions.append(session_data)
                progress.update(1)
    
    # Сохраняем в JSON Lines формате
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for session in sessions:
            f.write(json.dumps(session) + "\n")
    
    print(f"Сохранено {len(sessions)} сессий в {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--output", type=str, default="../data/training_sessions.jsonl")
    args = parser.parse_args()
    
    collect_training_data(args.episodes, args.output)