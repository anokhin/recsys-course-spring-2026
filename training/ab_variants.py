"""
Сравнение вариантов рекомендера на одном и том же потоке эпизодов симулятора.
Каждому эпизоду присваивается один и тот же пользователь и одна и та же инициация
(seed фиксирован), но рекомендеру подаётся разный (через replay).

Варианты:
  - sasrec_random_anchor: базовый SasRec-I2I (как в botify/i2i.py).
  - sasrec_first_anchor: всегда используем первый трек сессии как якорь.
  - als_i2i: используем наш ALS для top-i2i, без учёта сессии.
  - als_session: session-vector из ALS с подмесом user.
  - hybrid: SasRec кандидаты, rerank ALS-скором + artist penalty.
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
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "sim"))

from sim.envs.config import RecEnvConfigSchema  # noqa: E402
from sim.envs.env import RecEnv  # noqa: E402


def load_sasrec(path: Path) -> dict[int, list[int]]:
    r = {}
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            r[int(j["item_id"])] = [int(x) for x in j["recommendations"]]
    return r


def load_artists(path: Path):
    by_track: dict[int, str] = {}
    inv: dict[str, list[int]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            t = int(j["track"])
            a = str(j["artist"])
            by_track[t] = a
            inv[a].append(t)
    return by_track, {k: np.asarray(v, dtype=np.int64) for k, v in inv.items()}


class SasRecRandom:
    def __init__(self, i2i, rng):
        self.i2i, self.rng = i2i, rng
    def recommend(self, user, history, seen):
        if not history:
            return -1
        agg = defaultdict(float)
        for t, tm in history:
            agg[t] += tm
        anchors = list(agg.keys()); weights = [max(agg[t], 1e-3) for t in anchors]
        while anchors:
            a = self.rng.choices(anchors, weights=weights, k=1)[0]
            for t in self.i2i.get(a, []):
                if t not in seen:
                    return int(t)
            i = anchors.index(a); anchors.pop(i); weights.pop(i)
        return -1


class SasRecFirst:
    def __init__(self, i2i):
        self.i2i = i2i
    def recommend(self, user, history, seen):
        if not history:
            return -1
        anchor = history[0][0]
        for t in self.i2i.get(anchor, []):
            if t not in seen:
                return int(t)
        for anchor, _ in history:
            for t in self.i2i.get(anchor, []):
                if t not in seen:
                    return int(t)
        return -1


class ALSSession:
    def __init__(self, factors_path, track_artists, artist_tracks):
        d = np.load(factors_path)
        self.I = d["item_factors"]; self.U = d["user_factors"]
        self.ta = track_artists; self.at = artist_tracks
        self.n_items = self.I.shape[0]; self.n_users = self.U.shape[0]
    def recommend(self, user, history, seen):
        tail = history[:8]
        if not tail:
            return -1
        weights = np.array([max(tm, 0.05) for _, tm in tail], dtype=np.float32)
        weights = np.log1p(weights * 8.0)
        idx = np.array([t for t, _ in tail if 0 <= t < self.n_items], dtype=np.int64)
        if idx.size == 0:
            return -1
        sv = (weights[: idx.size, None] * self.I[idx]).sum(axis=0)
        if 0 <= user < self.n_users:
            sv = sv + 0.3 * self.U[user]
        n = float(np.linalg.norm(sv))
        if n < 1e-6:
            return -1
        sv = sv / n
        scores = self.I @ sv
        if seen:
            sidx = np.fromiter((t for t in seen if 0 <= t < self.n_items), dtype=np.int64)
            if sidx.size:
                scores[sidx] = -np.inf
        counts = Counter()
        for t, _ in history:
            a = self.ta.get(t)
            if a: counts[a] += 1
        for a, c in counts.items():
            tracks = self.at.get(a)
            if tracks is None or tracks.size == 0: continue
            scores[tracks] *= 0.5 ** c
        best = int(np.argmax(scores))
        return best if np.isfinite(scores[best]) else -1


class Hybrid:
    """SasRec top-N кандидатов со всех anchor-ов истории (взвешенно), rerank ALS+artist."""
    def __init__(self, i2i, factors_path, track_artists, artist_tracks, cand_n=80):
        self.i2i = i2i
        d = np.load(factors_path)
        self.I = d["item_factors"]; self.U = d["user_factors"]
        self.ta = track_artists; self.at = artist_tracks
        self.cand_n = cand_n
        self.n_items = self.I.shape[0]; self.n_users = self.U.shape[0]
    def recommend(self, user, history, seen):
        if not history:
            return -1
        # Собираем кандидатов как взвешенная сумма i2i-сиблигов
        agg = defaultdict(float)
        history_weights = {}
        for rank_h, (t, tm) in enumerate(history):
            w_h = max(tm, 0.1) / (1 + 0.25 * rank_h)  # свежие чуть важнее, но первый (с time=1) всегда доминирует
            history_weights[t] = w_h
            for pos, cand in enumerate(self.i2i.get(t, [])):
                if cand in seen:
                    continue
                # score от i2i: выше для более верхних позиций
                agg[cand] += w_h * (1.0 / math.log2(pos + 2.0))
        if not agg:
            return -1
        items = np.fromiter(agg.keys(), dtype=np.int64)
        scores = np.fromiter(agg.values(), dtype=np.float32)
        # ALS-rerank: учитываем direction сессии
        tail = history[:6]
        idx = np.array([t for t, _ in tail if 0 <= t < self.n_items], dtype=np.int64)
        if idx.size > 0:
            w = np.log1p(np.array([max(tm, 0.05) for _, tm in tail[:idx.size]], dtype=np.float32) * 8.0)
            sv = (w[:, None] * self.I[idx]).sum(axis=0)
            if 0 <= user < self.n_users:
                sv = sv + 0.25 * self.U[user]
            n = float(np.linalg.norm(sv))
            if n > 1e-6:
                sv = sv / n
                als = self.I[items] @ sv
                scores = scores + 0.5 * als  # добавляем ALS-скор к i2i-скору
        # artist penalty
        counts = Counter()
        for t, _ in history:
            a = self.ta.get(t)
            if a: counts[a] += 1
        if counts:
            artists = np.array([self.ta.get(int(t), "") for t in items])
            mults = np.ones(items.shape, dtype=np.float32)
            for i, a in enumerate(artists):
                if a in counts:
                    mults[i] = 0.5 ** counts[a]
            scores = scores * mults
        return int(items[int(np.argmax(scores))])


def run(variant_name, recommender, env, episodes, seed):
    env.seed(seed)
    rng_fallback = random.Random(seed + 100)
    total_times = []
    for ep in range(episodes):
        obs, _ = env.reset()
        user = int(obs["user"])
        history = [(int(obs["track"]), 1.0)]
        seen = {int(obs["track"])}
        total = 0.0
        done = False
        while not done:
            a = recommender.recommend(user, history, seen)
            if a < 0 or a in seen:
                a = rng_fallback.randrange(env.action_space.n)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            seen.add(a)
            history.append((a, float(r)))
            total += float(r)
        total_times.append(total)
    arr = np.asarray(total_times)
    print(f"{variant_name:25s}  mean={arr.mean():.3f}  sd={arr.std():.3f}  n={len(arr)}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=31312)
    args = ap.parse_args()

    sasrec = load_sasrec(REPO / "botify" / "data" / "sasrec_i2i.jsonl")
    track_artists, artist_tracks = load_artists(REPO / "botify" / "data" / "tracks.json")
    factors = REPO / "botify" / "data" / "personal_factors.npz"

    os.chdir(REPO / "sim")
    config = RecEnvConfigSchema().load(yaml.full_load(open("config/env.yml")))
    env = RecEnv(config)

    print(f"Running {args.episodes} episodes per variant, seed={args.seed}")
    run("sasrec_random_anchor", SasRecRandom(sasrec, random.Random(args.seed)), env, args.episodes, args.seed)
    run("sasrec_first_anchor", SasRecFirst(sasrec), env, args.episodes, args.seed)
    run("als_session", ALSSession(factors, track_artists, artist_tracks), env, args.episodes, args.seed)
    run("hybrid", Hybrid(sasrec, factors, track_artists, artist_tracks), env, args.episodes, args.seed)


if __name__ == "__main__":
    main()
