import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _iter_log_rows(data_dir: Path) -> Iterable[Dict]:
    for path in sorted(data_dir.glob("*/data.json")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _build_sessions(
    data_dir: Path,
    *,
    experiment_name: str = "HW2_AB",
    keep_treatments: Sequence[str] = ("C",),
) -> List[List[Tuple[int, float]]]:
    events_by_user = defaultdict(list)
    for row in _iter_log_rows(data_dir):
        if row.get("message") not in {"next", "last"}:
            continue

        experiments = row.get("experiments") or {}
        treatment = experiments.get(experiment_name)
        if treatment not in keep_treatments:
            continue

        events_by_user[int(row["user"])].append(row)

    sessions: List[List[Tuple[int, float]]] = []
    for _, events in events_by_user.items():
        events.sort(key=lambda x: int(x["timestamp"]))
        session: List[Tuple[int, float]] = []

        for e in events:
            track = int(e["track"])
            t = float(e.get("time", 0.0))
            session.append((track, t))

            if e["message"] == "last":
                if len(session) >= 2:
                    sessions.append(session)
                session = []

        if len(session) >= 2:
            sessions.append(session)

    return sessions


def load_i2i_pool(path: Path) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[int(row["item_id"])] = [int(x) for x in row["recommendations"]]
    return out


def load_track_meta(path: Path) -> Tuple[int, Dict[int, int], Dict[int, str]]:
    max_id = -1
    artist_ids: Dict[int, int] = {}
    genres: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tid = int(row["track"])
            max_id = max(max_id, tid)
            artist_ids[tid] = int(row.get("artist_id", -1))
            glist = row.get("genres") or []
            genres[tid] = str(glist[0]) if glist else "unknown"
    return max_id + 1, artist_ids, genres


def rank_in_list(items: List[int], target: int) -> int:
    try:
        return items.index(target)
    except ValueError:
        return 10**9


def build_dataset(
    sessions: List[List[Tuple[int, float]]],
    pool_by_anchor: Dict[int, List[int]],
    artist_ids: Dict[int, int],
    genres: Dict[int, str],
    global_counts: np.ndarray,
    negatives: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    X_rows: List[List[float]] = []
    y_vals: List[float] = []

    for session in sessions:
        tracks = [t for t, _ in session]
        times = [float(t) for _, t in session]
        for i in range(len(tracks) - 1):
            prev = tracks[i]
            nxt = tracks[i + 1]
            reward = max(0.0, times[i + 1])
            pool = pool_by_anchor.get(prev, [])
            if not pool:
                continue
            rpos = rank_in_list(pool, nxt)
            if rpos >= 10**8:
                continue

            def feats(prev_t: int, cand: int, rank: int) -> List[float]:
                ap = artist_ids.get(prev_t, -1)
                ac = artist_ids.get(cand, -2)
                gp = genres.get(prev_t, "")
                gc = genres.get(cand, "")
                return [
                    1.0 / (1.0 + rank),
                    min(rank, 99) / 99.0,
                    1.0 if rank < 5 else 0.0,
                    float(np.log1p(global_counts[cand])),
                    1.0 if ap == ac and ap >= 0 else 0.0,
                    1.0 if gp == gc and gp else 0.0,
                ]

            X_rows.append(feats(prev, nxt, rpos))
            y_vals.append(reward)

            negs = [c for c in pool if c != nxt]
            rng.shuffle(negs)
            for cand in negs[:negatives]:
                rp = rank_in_list(pool, cand)
                if rp >= 10**8:
                    continue
                X_rows.append(feats(prev, cand, rp))
                y_vals.append(0.0)

    return np.asarray(X_rows, dtype=np.float64), np.asarray(y_vals, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=str)
    parser.add_argument("--tracks-catalog", required=True, type=str)
    parser.add_argument("--candidates", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--negatives", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", type=str, default="HW2_AB")
    parser.add_argument("--treatments", type=str, default="C,T1")
    args = parser.parse_args()

    arms = tuple(x.strip() for x in args.treatments.split(",") if x.strip())
    sessions = _build_sessions(
        Path(args.data),
        experiment_name=args.experiment,
        keep_treatments=arms,
    )
    pool_by_anchor = load_i2i_pool(Path(args.candidates))
    num_items, artist_ids, genres = load_track_meta(Path(args.tracks_catalog))

    global_counts = np.zeros(num_items, dtype=np.float64)
    for sess in sessions:
        for tid, _ in sess:
            if 0 <= tid < num_items:
                global_counts[tid] += 1.0

    X, y = build_dataset(
        sessions=sessions,
        pool_by_anchor=pool_by_anchor,
        artist_ids=artist_ids,
        genres=genres,
        global_counts=global_counts,
        negatives=args.negatives,
        seed=args.seed,
    )
    if len(y) < 500:
        raise RuntimeError("too few training rows: %d" % len(y))

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=args.alpha, random_state=args.seed)),
        ]
    )
    pipe.fit(X, y)

    scaler = pipe.named_steps["scaler"]
    ridge = pipe.named_steps["ridge"]

    payload = {
        "model": "sklearn_ridge_standard_scaler",
        "candidate_pool": Path(args.candidates).name,
        "feature_names": [
            "inv_rank",
            "rank_norm",
            "top5",
            "log_global_pop",
            "same_artist",
            "same_genre",
        ],
        "coef": [float(c) for c in ridge.coef_],
        "intercept": float(ridge.intercept_),
        "mean": [float(m) for m in scaler.mean_],
        "scale": [float(s) if s > 1e-12 else 1.0 for s in scaler.scale_],
        "global_counts": global_counts.tolist(),
        "num_items": int(num_items),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("rows=%d saved=%s" % (len(y), out))


if __name__ == "__main__":
    main()
