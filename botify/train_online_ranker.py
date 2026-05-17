import argparse
import glob
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "hist_len",
    "avg_time",
    "last_time",
    "good_frac",
    "unique_artists",
    "cand_artist_count",
    "same_last_artist",
    "genre_jaccard_good",
    "mood_match_count",
    "year_abs_distance",
    "artist_fans_log",
    "sasrec_hit",
    "sasrec_best_rank_inv",
    "sasrec_weighted",
    "lightfm_hit",
    "lightfm_best_rank_inv",
    "lightfm_weighted",
    "dlrm_hit",
    "dlrm_best_rank_inv",
    "dlrm_weighted",
    "source_hits",
    "cand_global_mean_time",
    "cand_global_good_rate",
    "cand_global_skip_rate",
    "cand_global_log_count",
]


def load_i2i(path: Path):
    out = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[int(row["item_id"])] = [int(x) for x in row["recommendations"]]
    return out


def load_tracks(path: Path):
    meta = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            raw_year = row.get("year")
            try:
                year = int(raw_year) if raw_year not in (None, "", 0) else 0
            except (TypeError, ValueError):
                year = 0
            meta[int(row["track"])] = {
                "artist": row.get("artist"),
                "genres": set(row.get("genres") or []),
                "mood": row.get("mood"),
                "year": year,
                "fans": float(row.get("artist_fans") or 0.0),
            }
    return meta


def read_logs(patterns):
    frames = []
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            try:
                frame = pd.read_json(path, lines=True)
            except ValueError:
                continue
            if len(frame):
                frames.append(frame)
    if not frames:
        raise RuntimeError(f"No logs matched {patterns}")
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["user", "track", "time", "timestamp"])
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["track"] = data["track"].astype(int)
    data["time"] = data["time"].astype(float)
    return data.sort_values(["user", "timestamp"])


def global_stats(data):
    stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for track, tm in data[["track", "time"]].itertuples(index=False):
        s = stats[int(track)]
        s[0] += float(tm)
        s[1] += 1.0
        s[2] += 1.0 if tm >= 0.7 else 0.0
        s[3] += 1.0 if tm <= 0.1 else 0.0
    return stats


def candidate_set(history, seen, sources, anchor_window, per_source, max_candidates):
    cands = []
    added = set()
    for track, _ in reversed(history[-anchor_window:]):
        for name, table in sources.items():
            for cand in table.get(int(track), [])[:per_source]:
                cand = int(cand)
                if cand not in seen and cand not in added:
                    cands.append(cand)
                    added.add(cand)
                if len(cands) >= max_candidates:
                    return cands
    return cands


def features_for(history, candidates, sources, track_meta, stats, anchor_window):
    times = [tm for _, tm in history]
    avg_time = float(np.mean(times)) if times else 0.0
    last_time = float(times[-1]) if times else 0.0
    good_frac = float(np.mean([tm >= 0.7 for tm in times])) if times else 0.0

    artists = []
    liked_genres = set()
    liked_moods = Counter()
    years = []
    for track, tm in history:
        meta = track_meta.get(int(track))
        if not meta:
            continue
        artists.append(meta["artist"])
        if tm >= 0.5:
            liked_genres |= meta["genres"]
            liked_moods[meta["mood"]] += 1
        if meta["year"] > 0:
            years.append(meta["year"])
    artist_counts = Counter(artists)
    last_artist = track_meta.get(int(history[-1][0]), {}).get("artist") if history else None
    mean_year = float(np.mean(years)) if years else 0.0

    rank_tables = []
    for track, tm in history[-anchor_window:]:
        per_source = {}
        for name, table in sources.items():
            per_source[name] = {int(t): rank + 1 for rank, t in enumerate(table.get(int(track), []))}
        rank_tables.append((float(tm), per_source))

    X = np.zeros((len(candidates), len(FEATURES)), dtype=np.float32)
    name_to_idx = {name: idx for idx, name in enumerate(FEATURES)}

    for row_idx, cand in enumerate(candidates):
        meta = track_meta.get(int(cand), {})
        cand_genres = meta.get("genres", set())
        if cand_genres and liked_genres:
            genre_jaccard = len(cand_genres & liked_genres) / max(len(cand_genres | liked_genres), 1)
        else:
            genre_jaccard = 0.0
        year = meta.get("year", 0)
        year_dist = abs(year - mean_year) if year > 0 and mean_year > 0 else 0.0

        row = {
            "hist_len": len(history),
            "avg_time": avg_time,
            "last_time": last_time,
            "good_frac": good_frac,
            "unique_artists": len(set(artists)),
            "cand_artist_count": artist_counts.get(meta.get("artist"), 0),
            "same_last_artist": 1.0 if meta.get("artist") == last_artist and last_artist is not None else 0.0,
            "genre_jaccard_good": genre_jaccard,
            "mood_match_count": liked_moods.get(meta.get("mood"), 0),
            "year_abs_distance": year_dist / 50.0,
            "artist_fans_log": math.log1p(float(meta.get("fans", 0.0))),
        }

        source_hits = 0
        for source_name, prefix in [("sasrec", "sasrec"), ("lightfm", "lightfm"), ("dlrm", "dlrm")]:
            hit = 0
            best_inv = 0.0
            weighted = 0.0
            for anchor_time, tables in rank_tables:
                rank = tables.get(source_name, {}).get(int(cand))
                if rank is not None:
                    hit += 1
                    best_inv = max(best_inv, 1.0 / rank)
                    weighted += anchor_time / rank
            row[f"{prefix}_hit"] = hit
            row[f"{prefix}_best_rank_inv"] = best_inv
            row[f"{prefix}_weighted"] = weighted
            source_hits += 1 if hit > 0 else 0
        row["source_hits"] = source_hits

        st = stats.get(int(cand), [0.0, 0.0, 0.0, 0.0])
        cnt = max(st[1], 1.0)
        row["cand_global_mean_time"] = st[0] / cnt
        row["cand_global_good_rate"] = st[2] / cnt
        row["cand_global_skip_rate"] = st[3] / cnt
        row["cand_global_log_count"] = math.log1p(st[1])

        for name, value in row.items():
            X[row_idx, name_to_idx[name]] = float(value)
    return X


def build_dataset(data, sources, track_meta, stats, args):
    X_parts = []
    y_parts = []
    rng = np.random.default_rng(args.seed)
    examples = 0
    positives = 0

    for _, group in data.groupby("user", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        history = []
        for idx in range(len(group) - 1):
            row = group.iloc[idx]
            history.append((int(row["track"]), float(row["time"])))
            if row.get("message") != "next" or pd.isna(row.get("recommendation")):
                continue
            next_row = group.iloc[idx + 1]
            rec = int(row["recommendation"])
            if int(next_row["track"]) != rec:
                continue
            seen = {t for t, _ in history}
            candidates = candidate_set(history, seen, sources, args.anchor_window, args.per_source, args.max_candidates)
            if rec not in candidates and rec not in seen:
                candidates.append(rec)
            if len(candidates) < 2:
                continue

            y = np.zeros(len(candidates), dtype=np.int8)
            if float(next_row["time"]) >= args.good_time:
                try:
                    y[candidates.index(rec)] = 1
                    positives += 1
                except ValueError:
                    pass
            # Keep all states with positives, and a sample of bad-only states.
            if y.sum() == 0 and rng.random() > args.bad_state_sample:
                continue
            X_parts.append(features_for(history, candidates, sources, track_meta, stats, args.anchor_window))
            y_parts.append(y)
            examples += len(candidates)
            if examples >= args.max_rows:
                break
        if examples >= args.max_rows:
            break

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    print(f"rows={len(y)} positives={int(y.sum())} states_pos={positives} positive_rate={y.mean():.5f}")
    return X, y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", action="append", required=True)
    parser.add_argument("--tracks", type=Path, default=Path("data/tracks.json"))
    parser.add_argument("--sasrec", type=Path, default=Path("data/sasrec_i2i.jsonl"))
    parser.add_argument("--lightfm", type=Path, default=Path("data/lightfm_i2i.jsonl"))
    parser.add_argument("--dlrm", type=Path, default=Path("data/dlrm_sasrec_rerank_i2i.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/online_ranker_model.json"))
    parser.add_argument("--good-time", type=float, default=0.7)
    parser.add_argument("--bad-state-sample", type=float, default=0.25)
    parser.add_argument("--anchor-window", type=int, default=4)
    parser.add_argument("--per-source", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--max-rows", type=int, default=1_200_000)
    parser.add_argument("--seed", type=int, default=31337)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sources = {
        "sasrec": load_i2i(args.sasrec),
        "lightfm": load_i2i(args.lightfm),
        "dlrm": load_i2i(args.dlrm),
    }
    track_meta = load_tracks(args.tracks)
    data = read_logs(args.logs)
    stats = global_stats(data)
    X, y = build_dataset(data, sources, track_meta, stats, args)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=300, class_weight="balanced", C=0.5, solver="lbfgs")
    model.fit(Xs, y)
    payload = {
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "params": {
            "anchor_window": args.anchor_window,
            "per_source": args.per_source,
            "max_candidates": args.max_candidates,
        },
        "global_stats": {str(k): v for k, v in stats.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    print(f"saved: {args.output}")
