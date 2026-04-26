"""Train a LightGBM LambdaRank reranker on Botify control-arm logs.

For each user state in the control logs (where SasRec was the live recommender),
build a candidate pool from SasRec-I2I, LightFM-I2I anchors and HSTU user top-N,
then train a learning-to-rank model where the candidate that the user actually
accepted (with high listen time) is the positive.

Outputs:
  botify/data/learned_ranker.lgb     -- LightGBM model dump
  botify/data/learned_ranker_meta.json -- feature columns + hyperparameters
"""

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

FEATURES = [
    "hist_len",
    "avg_time",
    "last_time",
    "good_frac",
    "skip_frac",
    "unique_artists",
    "same_artist_last",
    "cand_artist_repeat",
    "genre_jaccard_liked",
    "mood_match_count",
    "year_dist",
    "artist_fans_log",
    "sasrec_hits",
    "sasrec_best_rr",
    "sasrec_weighted_rr",
    "lfm_hits",
    "lfm_best_rr",
    "lfm_weighted_rr",
    "source_agreement",
    "hstu_rank_inv",
    "hstu_present",
    "cand_global_mean_time",
    "cand_global_good_rate",
    "cand_global_log_count",
]

ANCHOR_WINDOW = 4
TOPK_PER_SOURCE = 10
MAX_CANDIDATES = 50
GOOD_TIME = 0.7


def load_jsonl_i2i(path: Path):
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["item_id"])] = [int(x) for x in row["recommendations"]]
    return out


def load_hstu_user_topn(path: Path):
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["user"])] = [int(x) for x in row["tracks"]]
    return out


def load_tracks(path: Path):
    meta = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                year = int(row.get("year") or 0)
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


def read_control_logs(patterns):
    rows = []
    for pat in patterns:
        for path in glob.glob(pat, recursive=True):
            with open(path) as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        ev = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("experiments", {}).get("HSTU") != "C":
                        continue
                    if ev.get("message") not in ("next", "last"):
                        continue
                    rows.append(ev)
    return rows


def build_sessions(rows):
    by_user = defaultdict(list)
    for ev in rows:
        by_user[int(ev["user"])].append(ev)
    sessions = []
    for user, events in by_user.items():
        events.sort(key=lambda e: int(e["timestamp"]))
        cur = []
        for ev in events:
            cur.append({
                "track": int(ev["track"]),
                "time": float(ev.get("time", 0.0)),
                "message": ev["message"],
                "recommendation": ev.get("recommendation"),
            })
            if ev["message"] == "last":
                if len(cur) >= 2:
                    sessions.append((user, cur))
                cur = []
        if len(cur) >= 2:
            sessions.append((user, cur))
    return sessions


def candidate_pool(history, seen, sasrec_i2i, lightfm_i2i):
    cands = []
    added = set()
    for track, _ in reversed(history[-ANCHOR_WINDOW:]):
        for src in (sasrec_i2i, lightfm_i2i):
            for cand in src.get(int(track), [])[:TOPK_PER_SOURCE]:
                cand = int(cand)
                if cand in seen or cand in added:
                    continue
                added.add(cand)
                cands.append(cand)
                if len(cands) >= MAX_CANDIDATES:
                    return cands
    return cands


def build_global_stats(rows):
    stats = defaultdict(lambda: [0.0, 0.0, 0.0])  # sum_time, count, good_count
    for ev in rows:
        if ev.get("message") != "next":
            continue
        try:
            track = int(ev["track"])
            t = float(ev["time"])
        except (TypeError, ValueError, KeyError):
            continue
        s = stats[track]
        s[0] += t
        s[1] += 1.0
        if t >= GOOD_TIME:
            s[2] += 1.0
    return stats


def features_for_state(history, prev_track, prev_time, candidates,
                       sasrec_i2i, lightfm_i2i, hstu_user_topn, user,
                       tracks_meta, global_stats):
    times = [tm for _, tm in history]
    avg_time = float(np.mean(times)) if times else 0.0
    last_time = float(prev_time)
    good_frac = float(np.mean([t >= 0.7 for t in times])) if times else 0.0
    skip_frac = float(np.mean([t < 0.2 for t in times])) if times else 0.0

    artists = []
    liked_genres = set()
    liked_moods = Counter()
    years = []
    for tr, tm in history:
        m = tracks_meta.get(int(tr))
        if not m:
            continue
        artists.append(m["artist"])
        if tm >= 0.5:
            liked_genres |= m["genres"]
            liked_moods[m["mood"]] += 1
        if m["year"] > 0:
            years.append(m["year"])
    artist_counts = Counter(artists)
    last_artist = tracks_meta.get(int(prev_track), {}).get("artist")
    mean_year = float(np.mean(years)) if years else 0.0

    rank_tables = []
    for tr, tm in history[-ANCHOR_WINDOW:]:
        sas = {int(t): r + 1 for r, t in enumerate(sasrec_i2i.get(int(tr), [])[:TOPK_PER_SOURCE])}
        lfm = {int(t): r + 1 for r, t in enumerate(lightfm_i2i.get(int(tr), [])[:TOPK_PER_SOURCE])}
        rank_tables.append((float(tm), sas, lfm))

    hstu_list = hstu_user_topn.get(int(user), [])
    hstu_rank = {int(t): r + 1 for r, t in enumerate(hstu_list)}

    rows = []
    for cand in candidates:
        m = tracks_meta.get(int(cand), {})
        cand_artist = m.get("artist")
        cand_genres = m.get("genres", set())
        if cand_genres and liked_genres:
            jacc = len(cand_genres & liked_genres) / max(len(cand_genres | liked_genres), 1)
        else:
            jacc = 0.0
        year = m.get("year", 0)
        year_dist = abs(year - mean_year) / 50.0 if year > 0 and mean_year > 0 else 0.0

        sas_hits = lfm_hits = 0
        sas_best = lfm_best = 0.0
        sas_w = lfm_w = 0.0
        agreement = 0
        for atime, sas, lfm in rank_tables:
            sr = sas.get(int(cand))
            lr = lfm.get(int(cand))
            if sr is not None:
                sas_hits += 1
                sas_best = max(sas_best, 1.0 / sr)
                sas_w += atime / sr
            if lr is not None:
                lfm_hits += 1
                lfm_best = max(lfm_best, 1.0 / lr)
                lfm_w += atime / lr
            if sr is not None and lr is not None:
                agreement += 1

        hr = hstu_rank.get(int(cand))
        hstu_inv = 1.0 / hr if hr is not None else 0.0
        hstu_pres = 1.0 if hr is not None else 0.0

        st = global_stats.get(int(cand), [0.0, 0.0, 0.0])
        cnt = max(st[1], 1.0)
        gmean = st[0] / cnt
        ggood = st[2] / cnt
        glog = math.log1p(st[1])

        rows.append({
            "hist_len": float(len(history)),
            "avg_time": avg_time,
            "last_time": last_time,
            "good_frac": good_frac,
            "skip_frac": skip_frac,
            "unique_artists": float(len(set(artists))),
            "same_artist_last": 1.0 if cand_artist == last_artist and cand_artist is not None else 0.0,
            "cand_artist_repeat": float(artist_counts.get(cand_artist, 0)),
            "genre_jaccard_liked": jacc,
            "mood_match_count": float(liked_moods.get(m.get("mood"), 0)),
            "year_dist": year_dist,
            "artist_fans_log": math.log1p(float(m.get("fans", 0.0))),
            "sasrec_hits": float(sas_hits),
            "sasrec_best_rr": sas_best,
            "sasrec_weighted_rr": sas_w,
            "lfm_hits": float(lfm_hits),
            "lfm_best_rr": lfm_best,
            "lfm_weighted_rr": lfm_w,
            "source_agreement": float(agreement),
            "hstu_rank_inv": hstu_inv,
            "hstu_present": hstu_pres,
            "cand_global_mean_time": gmean,
            "cand_global_good_rate": ggood,
            "cand_global_log_count": glog,
        })
    return rows


def build_dataset(sessions, sasrec_i2i, lightfm_i2i, hstu_user_topn,
                  tracks_meta, global_stats, max_states=None, seed=42):
    rng = np.random.default_rng(seed)
    X_rows = []
    y_rows = []
    group_sizes = []
    state_count = 0
    for user, session in sessions:
        for i in range(len(session) - 1):
            cur = session[i]
            nxt = session[i + 1]
            if cur["message"] != "next":
                continue
            true_next = cur.get("recommendation")
            if true_next is None:
                continue
            true_next = int(true_next)
            true_next_time = float(nxt["time"])
            if true_next_time < GOOD_TIME:
                continue  # only learn from "good" landings

            history = [(int(x["track"]), float(x["time"])) for x in session[: i + 1]]
            seen = {t for t, _ in history}
            cands = candidate_pool(history, seen, sasrec_i2i, lightfm_i2i)
            if true_next not in cands and true_next not in seen:
                cands = [true_next] + cands
            cands = [c for c in cands if c != int(cur["track"])]
            if len(cands) < 2 or true_next not in cands:
                continue

            rows = features_for_state(
                history,
                prev_track=int(cur["track"]),
                prev_time=float(cur["time"]),
                candidates=cands,
                sasrec_i2i=sasrec_i2i,
                lightfm_i2i=lightfm_i2i,
                hstu_user_topn=hstu_user_topn,
                user=user,
                tracks_meta=tracks_meta,
                global_stats=global_stats,
            )
            labels = [1 if c == true_next else 0 for c in cands]

            X_rows.extend(rows)
            y_rows.extend(labels)
            group_sizes.append(len(cands))
            state_count += 1
            if max_states is not None and state_count >= max_states:
                X = pd.DataFrame(X_rows)[FEATURES]
                return X, np.asarray(y_rows), np.asarray(group_sizes)
    X = pd.DataFrame(X_rows)[FEATURES]
    return X, np.asarray(y_rows), np.asarray(group_sizes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="+", default=[
        str(REPO_ROOT / "data" / "botify-recommender-1" / "data.json"),
        str(REPO_ROOT / "data" / "botify-recommender-2" / "data.json"),
    ])
    p.add_argument("--tracks", default=str(REPO_ROOT / "botify" / "data" / "tracks.json"))
    p.add_argument("--sasrec", default=str(REPO_ROOT / "botify" / "data" / "sasrec_i2i.jsonl"))
    p.add_argument("--lightfm", default=str(REPO_ROOT / "botify" / "data" / "lightfm_i2i.jsonl"))
    p.add_argument("--hstu", default=str(REPO_ROOT / "botify" / "data" / "hstu_recommendations.json"))
    p.add_argument("--model_out", default=str(REPO_ROOT / "botify" / "data" / "learned_ranker.lgb"))
    p.add_argument("--meta_out", default=str(REPO_ROOT / "botify" / "data" / "learned_ranker_meta.json"))
    p.add_argument("--max_states", type=int, default=20000)
    p.add_argument("--num_leaves", type=int, default=31)
    p.add_argument("--num_iter", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("Loading I2I tables...")
    sasrec = load_jsonl_i2i(Path(args.sasrec))
    lightfm = load_jsonl_i2i(Path(args.lightfm))
    hstu = load_hstu_user_topn(Path(args.hstu))
    tracks_meta = load_tracks(Path(args.tracks))
    print(f"  sasrec items: {len(sasrec)}  lightfm items: {len(lightfm)}  "
          f"hstu users: {len(hstu)}  tracks: {len(tracks_meta)}")

    print("Reading control logs...")
    rows = read_control_logs(args.logs)
    print(f"  control rows: {len(rows)}")

    print("Building global stats from logs...")
    global_stats = build_global_stats(rows)

    print("Building sessions...")
    sessions = build_sessions(rows)
    print(f"  sessions: {len(sessions)}")

    print("Building (state, candidate) dataset...")
    X, y, groups = build_dataset(
        sessions, sasrec, lightfm, hstu, tracks_meta, global_stats,
        max_states=args.max_states, seed=args.seed,
    )
    print(f"  rows: {len(y)}  states: {len(groups)}  positives: {int(y.sum())}")

    # train/valid split by state (group)
    rng = np.random.default_rng(args.seed)
    n_groups = len(groups)
    perm = rng.permutation(n_groups)
    n_valid = max(1, int(0.15 * n_groups))
    valid_g = set(perm[:n_valid].tolist())

    train_mask = np.zeros(len(y), dtype=bool)
    valid_mask = np.zeros(len(y), dtype=bool)
    cursor = 0
    for gi, gsize in enumerate(groups):
        sl = slice(cursor, cursor + gsize)
        if gi in valid_g:
            valid_mask[sl] = True
        else:
            train_mask[sl] = True
        cursor += gsize
    train_groups = [g for gi, g in enumerate(groups) if gi not in valid_g]
    valid_groups = [g for gi, g in enumerate(groups) if gi in valid_g]

    train_set = lgb.Dataset(X[train_mask].values, label=y[train_mask], group=train_groups)
    valid_set = lgb.Dataset(X[valid_mask].values, label=y[valid_mask], group=valid_groups,
                            reference=train_set)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        "learning_rate": args.lr,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": args.seed,
    }
    print("Training LightGBM lambdarank...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_iter,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model_out)
    meta = {
        "feature_cols": FEATURES,
        "params": {
            "anchor_window": ANCHOR_WINDOW,
            "topk_per_source": TOPK_PER_SOURCE,
            "max_candidates": MAX_CANDIDATES,
            "good_time": GOOD_TIME,
        },
        "global_stats": {str(k): v for k, v in global_stats.items()},
    }
    Path(args.meta_out).write_text(json.dumps(meta))
    print(f"saved model: {args.model_out}")
    print(f"saved meta:  {args.meta_out}  (entries: {len(meta['global_stats'])})")


if __name__ == "__main__":
    main()
