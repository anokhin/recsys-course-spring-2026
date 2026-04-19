import argparse
import cmd
import itertools
import json
import os
import os.path
import time
import urllib.request
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass, asdict


import numpy as np
import pandas as pd
import scipy.stats as ss
import tqdm
import yaml

from sim.agents import Recommender, DummyRecommender, RemoteRecommender
from sim.agents.console import ConsoleRecommender
from sim.envs import RecEnv
from sim.envs.config import RecEnvConfigSchema, RecEnvConfig

DUMMY = "dummy"
REMOTE = "remote"
CONSOLE = "console"


@dataclass
class EpisodeStats:
    day: int
    episode: int
    reward: float = 0.0
    steps: int = 0


def run_episode(day: int, episode: int, env: RecEnv, recommender: Recommender):
    observation, _ = env.reset()
    done = False
    reward = 1.0

    stats = EpisodeStats(day, episode)

    while not done:
        action = recommender.recommend(observation, reward, done)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        stats.reward += reward
        stats.steps += 1

    recommender.recommend(observation, reward, done)

    return stats


def run_episode_collect(
    day: int,
    episode: int,
    env: RecEnv,
    recommender: Recommender,
    recommender_name: str,
):
    """One transition per line: context (user, track) -> action -> reward and sim diagnostics."""
    observation, _ = env.reset()
    done = False
    reward = 1.0
    step = 0
    rows = []
    while not done:
        action = recommender.recommend(observation, reward, done)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        row = {
            "day": day,
            "episode": episode,
            "step": step,
            "user": int(observation["user"]),
            "context_track": int(observation["track"]),
            "recommended_track": int(action),
            "listen_time": float(reward),
            "episode_done": bool(done),
            "duplicate": bool(info.get("duplicate", False)),
            "affinity": info.get("affinity"),
            "recommended_artist": info.get("recommended_artist"),
            "recommender": recommender_name,
        }
        if "negative_tracks" in info:
            row["negative_tracks"] = info["negative_tracks"]
        rows.append(row)
        observation = next_observation
        step += 1

    recommender.recommend(observation, reward, done)
    return rows


def _make_recommender(env: RecEnv, recommender: str, config: RecEnvConfig):
    if recommender == DUMMY:
        return DummyRecommender(env.action_space), DUMMY
    if recommender == REMOTE:
        return RemoteRecommender(config.remote_recommender_config), REMOTE
    if recommender == CONSOLE:
        return ConsoleRecommender(config.remote_recommender_config), CONSOLE
    raise ValueError(f"Unknown recommender type: {recommender}")


def run_collect(args):
    config = RecEnvConfigSchema().load(yaml.full_load(open(args.config)))
    negatives = getattr(args, "negatives", 0)
    out_path = os.path.abspath(args.output)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    total = 0
    with open(out_path, "w", encoding="utf-8") as out_f, RecEnv(
        config, collect_negatives=negatives
    ) as env:
        env.seed(args.seed)
        recommender, recommender_name = _make_recommender(
            env, args.recommender, config
        )
        day = 1
        with recommender, tqdm.tqdm(total=args.episodes) as progress:
            for episode_id in range(args.episodes):
                for row in run_episode_collect(
                    day, episode_id, env, recommender, recommender_name
                ):
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total += 1
                progress.update(1)

    print(f"Wrote {total} transitions to {out_path}")
    return []


def run_experiment(
    day: int,
    env: RecEnv,
    episodes: int,
    recommender: str,
    config: RecEnvConfig,
    position=None,
):
    recommender, _ = _make_recommender(env, recommender, config)

    stats = []
    with recommender, tqdm.tqdm(total=episodes, position=position) as progress:
        for episode_id in range(episodes):
            stats.append(run_episode(day, episode_id, env, recommender))
            progress.update(1)
    return stats


def run_single(args):
    config = RecEnvConfigSchema().load(yaml.full_load(open(args.config)))

    stats = []

    with RecEnv(config) as env:
        env.seed(args.seed)

        day = 1
        while True:
            stats.extend(
                run_experiment(day, env, args.episodes, args.recommender, config)
            )

            time_control = TimeControl()
            time_control.cmdloop(
                f"End of day {day}. Would you like to start a new day?"
            )
            if time_control.done:
                break
            else:
                day += 1

    return stats


def _run_multi(process, args):
    config = RecEnvConfigSchema().load(yaml.full_load(open(args.config)))
    stats = []
    with RecEnv(config) as env:
        stats = run_experiment(
            1, env, args.episodes, REMOTE, config, position=process + 1
        )
    return stats


def run_multi(args):
    with ProcessPoolExecutor(args.processes) as executor:
        stats = executor.map(
            _run_multi, list(range(args.processes)), [args] * args.processes
        )
    return list(itertools.chain(*stats))


def download_data():
    print("Downloading simulator data...")
    print("1/3")
    embeddings_path = "data/embeddings.npy"
    if not os.path.exists(embeddings_path):
        urllib.request.urlretrieve(
            "https://download850.mediafire.com/82b8x5t778ggTu5HpNwCytXbTqKKiMLdfbr4O8gH8AOEc21OlayPpN_gc-hdN599KJ4ssGLsHKrjEvmBYKP8iudpUVngF2vzpbsDCbWtFtJZeAsfslnKRurGo1p_tzeqg571cUmo5cdUFPF19FqhapzOpznpxxXYBJVr1JsKs8KOJw/b42v7luqhwke12i/embeddings.npy",
            embeddings_path
        )

    print("2/3")
    tracks_path = "data/tracks.json"
    if not os.path.exists(tracks_path):
        urllib.request.urlretrieve(
            "https://download850.mediafire.com/kcjkxmqfdtuglJUswVNMI76Q-GFygr476CDaabM-Fx9jlHTWfZ2X9U7W-WktDNjVvTnGqt0qjHTCqF-2rOvxhOnk4uEEWYrEgH6ifvlih8sDvOYY8Hg2twurGosHM5vCxs6FslyNbp6EJmNandfMy-m5c76eUqtvsGiv3YmgLkO3Mw/busnvngp0jg9rer/tracks.json",
            tracks_path
        )

    print("3/3")
    users_path = "data/users.json"
    if not os.path.exists(users_path):
        urllib.request.urlretrieve(
            "https://download1323.mediafire.com/uxslpnd4o9pgoZTwVMFuHbbwKASqM_8vqN-mPdWZa8pSZVW1sZzeUZVvETpDaeRTS5C86MAG49-j2SM3CJ4jw1ZvffE8VM8nOv-5VnDQ859HXvIzntwQLqs56XaCbTFXmVban91JQOIpHcgRjtDVPl065Ui5PV03Pg4bBfX575tQdQ/x5vo04bjzwagy30/users.json",
                users_path
        )

    print("done")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        help="Path to environment config",
        type=str,
        default="config/env.yml",
    )

    parser.add_argument(
        "--episodes", help="Number of episodes in experiment", type=int, default=100
    )

    subparsers = parser.add_subparsers(help="modes of execution")

    single_parser = subparsers.add_parser(
        "single", help="Execute simulator in a single process"
    )
    single_parser.add_argument(
        "--recommender", choices=[DUMMY, REMOTE, CONSOLE], help="Recommender to use"
    )
    single_parser.add_argument(
        "--seed", help="Random seed for the env", type=int, default=42
    )
    single_parser.set_defaults(func=run_single)

    multi_parser = subparsers.add_parser(
        "multi", help="Execute simulator in multiple processes"
    )
    multi_parser.add_argument(
        "--processes",
        help="Number of simulations to execute in parallel",
        type=int,
        default=2,
    )
    multi_parser.set_defaults(func=run_multi)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Run episodes and export JSONL transitions for offline training",
    )
    collect_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output .jsonl (one object per line)",
    )
    collect_parser.add_argument(
        "--recommender",
        choices=[DUMMY, REMOTE, CONSOLE],
        default=DUMMY,
        help="dummy=uniform random (broad state coverage); remote=your botify service",
    )
    collect_parser.add_argument(
        "--seed", help="Random seed for the env", type=int, default=42
    )
    collect_parser.add_argument(
        "--negatives",
        type=int,
        default=10,
        help="Number of random unseen-track negatives per row (0 disables)",
    )
    collect_parser.set_defaults(func=run_collect)

    args = parser.parse_args()

    download_data()

    start = time.time()
    stats = args.func(args)
    print(f"Time: {int(time.time() - start)} seconds")

    if stats is None or len(stats) == 0:
        return

    result = (
        pd.DataFrame([asdict(s) for s in stats])
        .groupby("day")[["reward", "steps"]]
        .agg([np.mean, ss.sem])
    )
    print(f"## Experiment results summary\n\n{result.to_markdown()}")


class TimeControl(cmd.Cmd):
    prompt = "(y/n) "

    def __init__(self):
        super().__init__()
        self.done = False

    def do_y(self, arg):
        print("Moving to the next day!")
        return True

    def do_n(self, arg):
        print("Ending the simulation")
        self.done = True
        return True


if __name__ == "__main__":
    main()
