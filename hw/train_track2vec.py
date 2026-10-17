#!/usr/bin/env python3
"""Дообучение I2I на реальных логах botify (без gensim, без данных sim/).

После сбора логов: python hw/train_track2vec.py --logs data/hw2 --output botify/data/track2vec_i2i.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def load_track_artists(tracks_path: Path) -> dict[int, str]:
    track2artist = {}
    with tracks_path.open() as f:
        for line in f:
            row = json.loads(line)
            track2artist[int(row["track"])] = str(row["artist"])
    return track2artist


def load_sessions(log_dirs: list[Path], min_time: float) -> list[list[int]]:
    paths = []
    for d in log_dirs:
        paths.extend(glob.glob(str(d / "*/data.json")))
        paths.extend(glob.glob(str(d / "**/data.json"), recursive=True))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"Не найдены data.json в {log_dirs}")

    sessions: list[list[int]] = []
    for path in paths:
        by_user: dict[int, list] = defaultdict(list)
        with open(path) as f:
            for line in f:
                ev = json.loads(line)
                if "user" not in ev or "track" not in ev:
                    continue
                by_user[int(ev["user"])].append(ev)

        for events in by_user.values():
            events = sorted(events, key=lambda e: e.get("timestamp", 0))
            cur: list[int] = []
            for ev in events:
                if float(ev.get("time", 0.0)) >= min_time:
                    cur.append(int(ev["track"]))
                if ev.get("message") == "last":
                    if len(cur) >= 2:
                        sessions.append(cur)
                    cur = []
            if len(cur) >= 2:
                sessions.append(cur)
    return sessions


def diversify(neighbours, track2artist, top_k, max_per_artist):
    picked = []
    cnt = defaultdict(int)
    for tid in neighbours:
        a = track2artist.get(tid, "")
        if cnt[a] >= max_per_artist:
            continue
        picked.append(int(tid))
        cnt[a] += 1
        if len(picked) >= top_k:
            break
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--tracks", default="botify/data/tracks.json")
    parser.add_argument("--output", default="botify/data/track2vec_i2i.jsonl")
    parser.add_argument("--min-time", type=float, default=0.5)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-per-artist", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    track2artist = load_track_artists(Path(args.tracks))
    sessions = load_sessions([Path(p) for p in args.logs], min_time=args.min_time)
    print(f"Сессий: {len(sessions)}")
    if len(sessions) < 100:
        raise RuntimeError("Слишком мало сессий")

    vocab = sorted({t for s in sessions for t in s})
    index = {t: i for i, t in enumerate(vocab)}
    n = len(vocab)
    counts = lil_matrix((n, n), dtype=np.float32)
    for seq in sessions:
        for i, a in enumerate(seq):
            ia = index[a]
            for j in range(i + 1, min(i + args.window + 1, len(seq))):
                ib = index[seq[j]]
                if ia == ib:
                    continue
                counts[ia, ib] += 1.0
                counts[ib, ia] += 1.0

    X = counts.tocsr()
    X.data = np.log1p(X.data)
    emb = TruncatedSVD(
        n_components=min(args.dim, n - 1), random_state=args.seed
    ).fit_transform(X)
    emb = normalize(emb.astype(np.float32))

    out = Path(args.output)
    pool = max(200, args.top_k * 6)
    with out.open("w") as f:
        for i, tid in enumerate(vocab):
            sims = emb[i] @ emb.T
            sims[i] = -1.0
            cand = np.argpartition(sims, -pool)[-pool:]
            cand = cand[np.argsort(sims[cand])[::-1]]
            neighbours = diversify(
                [vocab[j] for j in cand],
                track2artist,
                args.top_k,
                args.max_per_artist,
            )
            f.write(
                json.dumps({"item_id": int(tid), "recommendations": neighbours}) + "\n"
            )
    print(f"Saved → {out} ({len(vocab)} items)")


if __name__ == "__main__":
    main()
