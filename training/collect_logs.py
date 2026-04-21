"""
Собирает обучающие логи из симулятора без docker.
Для каждого эпизода запускает сессию RecEnv, рекомендует треки через смесь
(1) случайного выбора (исследование) и (2) SasRec-I2I (эксплуатация, чтобы
сессии жили дольше и набирали данные). Пишет (user, track, time, step, session_id)
в JSONL.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "sim"))

from sim.envs.config import RecEnvConfigSchema  # noqa: E402
from sim.envs.env import RecEnv  # noqa: E402


def load_sasrec_i2i(path: Path) -> dict[int, list[int]]:
    i2i: dict[int, list[int]] = {}
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            i2i[int(j["item_id"])] = [int(t) for t in j["recommendations"]]
    return i2i


class MixedRecommender:
    """С вероятностью eps — случайный трек, иначе первый новый из SasRec-I2I по последнему прослушанному."""

    def __init__(self, i2i: dict[int, list[int]], n_tracks: int, eps: float, rng: random.Random):
        self.i2i = i2i
        self.n_tracks = n_tracks
        self.eps = eps
        self.rng = rng

    def pick(self, prev_track: int, seen: set[int]) -> int:
        if self.rng.random() < self.eps:
            for _ in range(16):
                t = self.rng.randrange(self.n_tracks)
                if t not in seen:
                    return t
            return self.rng.randrange(self.n_tracks)
        cand = self.i2i.get(prev_track)
        if cand:
            for t in cand:
                if t not in seen:
                    return t
        for _ in range(16):
            t = self.rng.randrange(self.n_tracks)
            if t not in seen:
                return t
        return self.rng.randrange(self.n_tracks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60000)
    ap.add_argument("--eps", type=float, default=0.35, help="Доля случайных рекомендаций для exploration")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=str(HERE / "sessions.jsonl"))
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    os.chdir(REPO / "sim")
    config = RecEnvConfigSchema().load(yaml.full_load(open("config/env.yml")))

    i2i = load_sasrec_i2i(REPO / "botify" / "data" / "sasrec_i2i.jsonl")
    rng = random.Random(args.seed)

    env = RecEnv(config)
    env.seed(args.seed)

    rec = MixedRecommender(i2i, n_tracks=env.action_space.n, eps=args.eps, rng=rng)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    total_time = 0.0
    with open(out_path, "w") as f:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            seen: set[int] = {obs["track"]}
            step = 0
            f.write(json.dumps({
                "episode": ep,
                "user": int(obs["user"]),
                "track": int(obs["track"]),
                "time": 1.0,
                "step": 0,
                "is_first": True,
            }) + "\n")
            prev_track = obs["track"]
            while not done:
                action = rec.pick(prev_track, seen)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                step += 1
                total_steps += 1
                total_time += float(reward)
                seen.add(action)
                f.write(json.dumps({
                    "episode": ep,
                    "user": int(obs["user"]),
                    "track": int(action),
                    "time": float(reward),
                    "step": step,
                    "is_first": False,
                }) + "\n")
                prev_track = action
            if (ep + 1) % 5000 == 0:
                print(f"  episode {ep + 1}/{args.episodes} | avg_steps_per_ep={total_steps / (ep + 1):.2f} | avg_time={total_time / max(total_steps, 1):.3f}")

    print(f"Done. steps={total_steps} avg_steps={total_steps / args.episodes:.2f} avg_time={total_time / max(total_steps, 1):.3f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
