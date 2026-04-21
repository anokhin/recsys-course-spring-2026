"""
In-process A/B: прогоняет симулятор, расщепляя пользователей 50/50 между
SasRec-I2I (C) и Personalized (T1), и считает mean_time_per_session аналогично
analyze_ab.py. Нужен, чтобы быстро проверить выигрыш до поднятия docker.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as ss
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "sim"))

from sim.envs.config import RecEnvConfigSchema  # noqa: E402
from sim.envs.env import RecEnv  # noqa: E402


def load_jsonl_int_list(path: Path, key_item: str, key_list: str) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            out[int(j[key_item])] = [int(x) for x in j[key_list]]
    return out


def load_track_artists(tracks_path: Path) -> tuple[dict[int, str], dict[str, np.ndarray]]:
    artists: dict[int, str] = {}
    inv: dict[str, list[int]] = defaultdict(list)
    with open(tracks_path) as f:
        for line in f:
            j = json.loads(line)
            t = int(j["track"])
            a = str(j["artist"])
            artists[t] = a
            inv[a].append(t)
    return artists, {a: np.asarray(v, dtype=np.int64) for a, v in inv.items()}


class SasRecI2I:
    def __init__(self, i2i: dict[int, list[int]], rng: random.Random):
        self.i2i = i2i
        self.rng = rng

    def recommend(self, history, seen):
        if not history:
            return -1
        agg = defaultdict(float)
        for t, tm in history:
            agg[t] += tm
        anchors = list(agg.keys())
        weights = [max(agg[t], 1e-3) for t in anchors]
        while anchors:
            anchor = self.rng.choices(anchors, weights=weights, k=1)[0]
            cand = self.i2i.get(anchor)
            if cand:
                for t in cand:
                    if t not in seen:
                        return int(t)
            i = anchors.index(anchor)
            anchors.pop(i)
            weights.pop(i)
        return -1


class PersonalizedLocal:
    ARTIST_DISCOUNT = 0.5
    USER_PRIOR_WEIGHT = 0.25
    HISTORY_DEPTH = 6

    def __init__(self, factors_path: Path, track_artists: dict[int, str], artist_tracks: dict[str, np.ndarray], fallback: SasRecI2I):
        data = np.load(factors_path)
        self.item_factors = data["item_factors"].astype(np.float32)
        self.user_factors = data["user_factors"].astype(np.float32)
        self.track_artists = track_artists
        self.artist_tracks = artist_tracks
        self.fallback = fallback
        self.n_tracks = self.item_factors.shape[0]
        self.n_users = self.user_factors.shape[0]

    def recommend(self, user, history, seen):
        session_vec = self._session_vector(user, history)
        if session_vec is None:
            return self.fallback.recommend(history, seen)
        scores = self.item_factors @ session_vec
        if seen:
            idx = np.fromiter((t for t in seen if 0 <= t < self.n_tracks), dtype=np.int64)
            if idx.size:
                scores[idx] = -np.inf
        artist_counts = Counter()
        for t, _ in history:
            a = self.track_artists.get(t)
            if a:
                artist_counts[a] += 1
        for a, c in artist_counts.items():
            if c <= 0:
                continue
            tracks = self.artist_tracks.get(a)
            if tracks is None or tracks.size == 0:
                continue
            scores[tracks] *= self.ARTIST_DISCOUNT ** c
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            return self.fallback.recommend(history, seen)
        return best

    def _session_vector(self, user, history):
        tail = history[: self.HISTORY_DEPTH]
        valid = [(t, tm) for t, tm in tail if 0 <= t < self.n_tracks]
        if not valid:
            if 0 <= user < self.n_users:
                v = self.user_factors[user]
                n = float(np.linalg.norm(v))
                return v / n if n > 1e-6 else None
            return None
        w = np.log1p(np.array([max(float(tm), 0.05) for _, tm in valid], dtype=np.float32) * 8.0)
        idx = np.array([t for t, _ in valid], dtype=np.int64)
        sv = (w[:, None] * self.item_factors[idx]).sum(axis=0)
        if 0 <= user < self.n_users:
            sv = sv + self.USER_PRIOR_WEIGHT * self.user_factors[user]
        n = float(np.linalg.norm(sv))
        return (sv / n).astype(np.float32) if n > 1e-6 else None


def assign(user: int) -> str:
    import mmh3
    seed = mmh3.hash("LOCAL_AB", signed=False) & 0xFFFFFFFF
    return "C" if (mmh3.hash(str(user), seed, False) % 2) == 0 else "T1"


def compute_effect(per_user: dict[str, list[tuple[int, float]]]):
    rows = []
    for treat, data in per_user.items():
        for u, t in data:
            rows.append({"treatment": treat, "user": u, "mean_time_per_session": t})
    df = pd.DataFrame(rows)
    agg = df.groupby("treatment")["mean_time_per_session"].agg(["count", "mean", "var"])
    print(agg)
    if "C" not in agg.index or "T1" not in agg.index:
        return
    n0, m0, v0 = agg.loc["C"]
    n1, m1, v1 = agg.loc["T1"]
    effect = m1 - m0
    dof_num = (v0 / n0 + v1 / n1) ** 2
    dof_den = v0 ** 2 / n0 ** 2 / (n0 - 1) + v1 ** 2 / n1 ** 2 / (n1 - 1)
    dof = dof_num / dof_den
    conf = ss.t.ppf(0.975, dof) * math.sqrt(v0 / n0 + v1 / n1)
    pct = effect / m0 * 100
    low = (effect - conf) / m0 * 100
    high = (effect + conf) / m0 * 100
    sig = (effect - conf) * (effect + conf) > 0
    print(f"\nC  mean={m0:.3f}  n={int(n0)}")
    print(f"T1 mean={m1:.3f}  n={int(n1)}")
    print(f"effect = {pct:+.2f}%  CI95 = [{low:+.2f}%; {high:+.2f}%]  significant={sig}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=31312)
    args = ap.parse_args()

    print("Loading artifacts...")
    sasrec_i2i = load_jsonl_int_list(REPO / "botify" / "data" / "sasrec_i2i.jsonl", "item_id", "recommendations")
    track_artists, artist_tracks = load_track_artists(REPO / "botify" / "data" / "tracks.json")

    os.chdir(REPO / "sim")
    config = RecEnvConfigSchema().load(yaml.full_load(open("config/env.yml")))
    env = RecEnv(config)
    env.seed(args.seed)

    rng_c = random.Random(args.seed + 1)
    rng_t = random.Random(args.seed + 2)

    sas = SasRecI2I(sasrec_i2i, rng_c)
    personal = PersonalizedLocal(
        REPO / "botify" / "data" / "personal_factors.npz",
        track_artists,
        artist_tracks,
        SasRecI2I(sasrec_i2i, rng_t),
    )

    per_user: dict[str, list[tuple[int, float]]] = {"C": [], "T1": []}
    for ep in range(args.episodes):
        obs, _ = env.reset()
        user = int(obs["user"])
        treat = assign(user)

        history = [(int(obs["track"]), 1.0)]
        seen = {int(obs["track"])}
        total_time = 0.0
        done = False
        while not done:
            if treat == "C":
                action = sas.recommend(history, seen)
            else:
                action = personal.recommend(user, history, seen)
            if action < 0:
                action = rng_c.randrange(env.action_space.n)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            seen.add(action)
            history.append((action, float(reward)))
            total_time += float(reward)
        per_user[treat].append((user, total_time))
        if (ep + 1) % 2000 == 0:
            c = np.mean([t for _, t in per_user["C"]]) if per_user["C"] else 0
            t = np.mean([t for _, t in per_user["T1"]]) if per_user["T1"] else 0
            print(f"  ep {ep+1}/{args.episodes}  C={c:.3f} (n={len(per_user['C'])})  T1={t:.3f} (n={len(per_user['T1'])})")

    compute_effect(per_user)


if __name__ == "__main__":
    main()
