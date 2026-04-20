"""
Collect training data by running the simulator against an ExplorationRecommender.

Usage:
  .venv/bin/python script/collect_training_data.py --episodes 5000 --seed 42 \
    --out data/train_raw --epsilon 0.3
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, env=None, cwd=None, shell=False):
    printable = cmd if shell else " ".join(cmd)
    print(f"$ {printable}")
    r = subprocess.run(cmd, env=env, cwd=cwd, shell=shell)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/train_raw")
    ap.add_argument("--epsilon", type=float, default=0.3)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    out = (repo / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    env_file = repo / "botify" / ".env.collect"
    env_file.write_text(f"BOTIFY_COLLECT=1\nBOTIFY_EPSILON={args.epsilon}\n")
    compose_cwd = str(repo / "botify")

    try:
        run(["docker", "compose", "--env-file", str(env_file),
             "down", "-v", "--remove-orphans"], cwd=compose_cwd)
        run(["docker", "compose", "--env-file", str(env_file),
             "up", "-d", "--build", "--force-recreate", "--scale", "recommender=2"],
            cwd=compose_cwd)
        print("Waiting 20s for gunicorn boot...")
        time.sleep(20)

        venv_py = repo / ".venv" / "bin" / "python"
        if not venv_py.exists():
            venv_py = Path(sys.executable)

        sim_cmd = (
            f'echo "n" | "{venv_py}" -m sim.run '
            f'--episodes {args.episodes} --config config/env.yml '
            f'single --recommender remote --seed {args.seed}'
        )
        run(sim_cmd, cwd=str(repo / "sim"), shell=True)

        run([str(venv_py), str(repo / "script" / "dataclient.py"),
             "--recommender", "2", "log2local", str(out)])
    finally:
        run(["docker", "compose", "--env-file", str(env_file),
             "down", "-v", "--remove-orphans"], cwd=compose_cwd)
        if env_file.exists():
            env_file.unlink()


if __name__ == "__main__":
    main()
