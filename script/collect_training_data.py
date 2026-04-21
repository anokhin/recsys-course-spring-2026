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


def run(cmd, env=None, cwd=None, shell=False, input_text=None):
    printable = cmd if shell else " ".join(cmd)
    print(f"$ {printable}", flush=True)
    r = subprocess.run(cmd, env=env, cwd=cwd, shell=shell,
                       input=input_text.encode() if input_text else None)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/train_raw")
    ap.add_argument("--epsilon", type=float, default=0.3)
    args = ap.parse_args()

    import shutil
    repo = Path(__file__).resolve().parent.parent
    out = (repo / args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    env_file = repo / "botify" / ".env.collect"
    env_file.write_text(f"BOTIFY_COLLECT=1\nBOTIFY_EPSILON={args.epsilon}\n")
    compose_cwd = str(repo / "botify")

    def copy_logs():
        for i in (1, 2):
            dest = out / f"botify-recommender-{i}"
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run(["docker", "cp",
                            f"botify-recommender-{i}:/app/botify/log/.",
                            str(dest)])

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

        sim_env = dict(os.environ)
        sim_env["PYTHONPATH"] = str(repo / "sim") + os.pathsep + sim_env.get("PYTHONPATH", "")
        sim_rc = subprocess.run(
            [str(venv_py), "-m", "sim.run",
             "--episodes", str(args.episodes),
             "--config", "config/env.yml",
             "single", "--recommender", "remote",
             "--seed", str(args.seed)],
            cwd=str(repo / "sim"), env=sim_env,
            input=b"n\n").returncode
        if sim_rc != 0:
            print(f"WARNING: sim exited {sim_rc}; still copying logs")
        copy_logs()
    finally:
        run(["docker", "compose", "--env-file", str(env_file),
             "down", "-v", "--remove-orphans"], cwd=compose_cwd)
        if env_file.exists():
            env_file.unlink()


if __name__ == "__main__":
    main()
