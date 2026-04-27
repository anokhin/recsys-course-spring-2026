#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.run import run_episode, TimeControl
from sim.envs import RecEnv
from sim.envs.config import RecEnvConfigSchema
from sim.agents.remote import RemoteRecommender
import yaml
import tqdm


def collect_data(episodes: int, output_file: str):
    config = RecEnvConfigSchema().load(yaml.full_load(open("sim/config/env.yml")))
    
    sessions = []
    
    with RecEnv(config) as env:
        env.seed(42)
        recommender = RemoteRecommender(config.remote_recommender_config)
        
        with recommender, tqdm.tqdm(total=episodes) as progress:
            for episode_id in range(episodes):
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
                    
                    event = {
                        "track": int(action),
                        "user": observation.get("user", 0),
                    }
                    
                    observation, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    event["listen_percent"] = float(reward)
                    session_data["events"].append(event)
                
                recommender.recommend(observation, reward, done)
                sessions.append(session_data)
                progress.update(1)
    
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
    parser.add_argument("--output", type=str, default="data/sasrec_sessions.jsonl")
    args = parser.parse_args()
    
    collect_data(args.episodes, args.output)