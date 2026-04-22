from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent
BOTIFY_ROOT = REPO_ROOT / "botify"
SIM_ROOT = REPO_ROOT / "sim"
SIM_DATA_ROOT = SIM_ROOT / "local_data"

# The simulator package expects imports like `sim.envs`, so we add the
# repository-local package roots explicitly instead of relying on docker/setup.py.
for path in (REPO_ROOT, BOTIFY_ROOT, SIM_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import numpy as np
import tqdm
import yaml

from botify.experiment import Experiments, Treatment
from botify.recommenders.i2i import I2IRecommender
from botify.recommenders.random import Random
from botify.recommenders.session_semantic import SessionSemanticRecommender
from botify.track import Catalog
from sim.envs import RecEnv
from sim.envs.config import RecEnvConfigSchema


class _Logger:
    def info(self, *args, **kwargs):
        pass


class _App:
    logger = _Logger()

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}


class InMemoryRedis:
    def __init__(self):
        self.store: Dict[object, object] = {}

    def set(self, key, value):
        self.store[key] = value

    def get(self, key):
        return self.store.get(key)

    def lpush(self, key, value):
        self.store.setdefault(key, [])
        self.store[key].insert(0, value)

    def ltrim(self, key, start, end):
        items = self.store.get(key, [])
        if end == -1:
            self.store[key] = items[start:]
        else:
            self.store[key] = items[start : end + 1]

    def lrange(self, key, start, end):
        items = self.store.get(key, [])
        if end == -1:
            return items[start:]
        return items[start : end + 1]

    def randomkey(self):
        if not self.store:
            raise KeyError("randomkey called on empty redis")
        return random.choice(list(self.store.keys()))


@dataclass
class LocalABAgent:
    history_redis: InMemoryRedis
    control: I2IRecommender
    treatment: SessionSemanticRecommender
    log_fp: object
    experiments = Experiments()

    def _persist_history(self, user: int, track: int, track_time: float):
        key = f"user:{user}:listens"
        payload = json.dumps({"track": int(track), "time": float(track_time)})
        self.history_redis.lpush(key, payload)
        self.history_redis.ltrim(key, 0, 9)

    def _log_event(
        self,
        message: str,
        user: int,
        track: int,
        track_time: float,
        latency: float,
        recommendation: Optional[int] = None,
    ):
        row = {
            "message": message,
            "timestamp": int(time.time() * 1000),
            "user": int(user),
            "track": int(track),
            "time": float(track_time),
            "latency": float(latency),
            "recommendation": None if recommendation is None else int(recommendation),
            "experiments": {
                self.experiments.SESSION_SEMANTIC.name: self.experiments.SESSION_SEMANTIC.assign(int(user)).name
            },
        }
        self.log_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    def recommend(self, observation: Dict[str, int], reward: float, done: bool):
        user = int(observation["user"])
        track = int(observation["track"])
        self._persist_history(user, track, reward)

        start = time.perf_counter()
        if done:
            if self.experiments.SESSION_SEMANTIC.assign(user) != Treatment.C:
                self.treatment.observe(user, track, reward)
                self.treatment.finish_session(user)
            latency = time.perf_counter() - start
            self._log_event("last", user, track, reward, latency, recommendation=None)
            return track

        treatment = self.experiments.SESSION_SEMANTIC.assign(user)
        if treatment == Treatment.C:
            recommender = self.control
        else:
            self.treatment.observe(user, track, reward)
            recommender = self.treatment
        recommendation = int(recommender.recommend_next(user, track, reward))
        latency = time.perf_counter() - start
        self._log_event("next", user, track, reward, latency, recommendation=recommendation)
        return recommendation


def _resolve_sim_data_root() -> Path:
    for root in (SIM_DATA_ROOT, SIM_ROOT / "data"):
        required = [root / "embeddings.npy", root / "tracks.json", root / "users.json"]
        if not all(path.exists() for path in required):
            continue
        if all(not path.read_bytes()[:32].startswith(b"version https://git-lfs.github.com/spec") for path in required):
            return root
    raise RuntimeError(
        "Simulator data not found. Put real files into sim/local_data/ or replace LFS pointers in sim/data/."
    )


def _assert_real_sim_data(data_root: Path):
    required = [
        data_root / "embeddings.npy",
        data_root / "tracks.json",
        data_root / "users.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing simulator data: {path}")
        head = path.read_bytes()[:32]
        if head.startswith(b"version https://git-lfs.github.com/spec"):
            raise RuntimeError(
                f"{path} is still a Git LFS pointer, not the real file. Download real simulator data first."
            )


def build_agent(log_path: Path) -> LocalABAgent:
    app = _App()
    catalog = Catalog(app).load(str(BOTIFY_ROOT / "data" / "tracks.json"))

    tracks_redis = InMemoryRedis()
    history_redis = InMemoryRedis()
    sasrec_redis = InMemoryRedis()
    lfm_redis = InMemoryRedis()
    hstu_redis = InMemoryRedis()

    catalog.upload_tracks(tracks_redis)
    catalog.app = _App(
        {
            "RECOMMENDATIONS_SASREC_FILE_PATH": str(BOTIFY_ROOT / "data" / "sasrec_i2i.jsonl"),
            "RECOMMENDATIONS_LFM_FILE_PATH": str(BOTIFY_ROOT / "data" / "lightfm_i2i.jsonl"),
            "RECOMMENDATIONS_HSTU_FILE_PATH": str(BOTIFY_ROOT / "data" / "hstu_recommendations.json"),
        }
    )
    catalog.upload_recommendations(
        sasrec_redis,
        "RECOMMENDATIONS_SASREC_FILE_PATH",
        key_object="item_id",
        key_recommendations="recommendations",
    )
    catalog.upload_recommendations(
        lfm_redis,
        "RECOMMENDATIONS_LFM_FILE_PATH",
        key_object="item_id",
        key_recommendations="recommendations",
    )
    catalog.upload_recommendations(
        hstu_redis,
        "RECOMMENDATIONS_HSTU_FILE_PATH",
        key_object="user",
        key_recommendations="tracks",
    )

    random_recommender = Random(tracks_redis)
    control = I2IRecommender(history_redis, sasrec_redis, random_recommender)
    treatment = SessionSemanticRecommender(
        history_redis,
        catalog,
        BOTIFY_ROOT / "data" / "session_semantic_embeddings.npz",
        control,
        i2i_redis=sasrec_redis,
        lfm_i2i_redis=lfm_redis,
        hstu_redis=hstu_redis,
        artist_penalty=0.22,
        session_profile_weight=0.78,
        prototype_weight=0.22,
        hstu_prior_weight=0.0,
        i2i_bonus=0.07,
        lfm_bonus=0.0,
        hstu_bonus=0.0,
        semantic_gate=0.14,
        min_margin=0.010,
    )

    log_fp = log_path.open("w", encoding="utf-8")
    return LocalABAgent(history_redis=history_redis, control=control, treatment=treatment, log_fp=log_fp)


def run_validation(seed: int, episodes: int, out_dir: Path):
    data_root = _resolve_sim_data_root()
    _assert_real_sim_data(data_root)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    run_dir = out_dir / "botify-recommender-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "data.json"

    random.seed(seed)
    np.random.seed(seed)

    config = RecEnvConfigSchema().load(yaml.full_load((SIM_ROOT / "config" / "env.yml").read_text()))
    config.track_catalog_config.tracks_path = str(data_root / Path(config.track_catalog_config.tracks_path).name)
    config.track_catalog_config.tracks_embeddings_path = str(
        data_root / Path(config.track_catalog_config.tracks_embeddings_path).name
    )
    config.user_catalog_config.user_catalog_path = str(data_root / Path(config.user_catalog_config.user_catalog_path).name)
    agent = build_agent(log_path)

    try:
        with RecEnv(config) as env:
            env.seed(seed)
            for _ in tqdm.trange(episodes, desc="offline-ab", unit="ep"):
                observation, _ = env.reset()
                done = False
                reward = 1.0
                while not done:
                    action = agent.recommend(observation, reward, done=False)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                agent.recommend(observation, reward, done=True)
    finally:
        agent.log_fp.close()

    ab_result_path = out_dir / "ab_result.json"
    subprocess.run(
        [
            str(REPO_ROOT.parent / ".venv" / "bin" / "python"),
            str(REPO_ROOT / "analyze_ab.py"),
            "--data",
            str(out_dir),
            "--output",
            str(ab_result_path),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    return ab_result_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=31312)
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--output", type=str, default="offline_validation")
    args = parser.parse_args()

    ab_path = run_validation(args.seed, args.episodes, REPO_ROOT / args.output)
    ab = json.loads(ab_path.read_text())
    effect = next(e for e in ab["all_effects"] if e["metric"] == "mean_time_per_session")
    print(
        "mean_time_per_session",
        f"control={effect['control_mean']:.4f}",
        f"treatment={effect['treatment_mean']:.4f}",
        f"effect={effect['effect_pct']:+.2f}%",
        f"CI=[{effect['lower_pct']:+.2f}%; {effect['upper_pct']:+.2f}%]",
        f"significant={effect['significant']}",
    )


if __name__ == "__main__":
    main()
