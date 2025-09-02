import argparse
import cmd
import itertools
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
    observation = env.reset()
    done = False
    reward = 1.0

    stats = EpisodeStats(day, episode)

    while not done:
        action = recommender.recommend(observation, reward, done)
        observation, reward, done, info = env.step(action)
        stats.reward += reward
        stats.steps += 1

    recommender.recommend(observation, reward, done)

    return stats


def run_experiment(
    day: int,
    env: RecEnv,
    episodes: int,
    recommender: str,
    config: RecEnvConfig,
    position=None,
):
    if recommender == DUMMY:
        recommender = DummyRecommender(env.action_space)
    elif recommender == REMOTE:
        recommender = RemoteRecommender(config.remote_recommender_config)
    elif recommender == CONSOLE:
        recommender = ConsoleRecommender(config.remote_recommender_config)
    else:
        raise ValueError(f"Unknown recommender type: {recommender}")

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

    args = parser.parse_args()

    download_data()

    start = time.time()
    stats = args.func(args)
    print(f"Time: {int(time.time() - start)} seconds")

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
