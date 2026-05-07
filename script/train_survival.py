import argparse
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


SKIP_THRESHOLD = 0.5
ANCHOR_MIN_TIME = 0.3
EPS_DURATION = 0.01
COX_PENALIZER = 0.01
DETERMINISTIC_HEAD = 20
GUMBEL_TAU = 0.0
TRAIN_SAMPLE_CAP = 200_000
FEATURE_COLS = [
    "pmi",
    "beta_binom_completion",
    "dot_score",
    "same_artist",
    "pop_score",
    "cand_skip_rate",
]


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", required=True)
    p.add_argument("--candidates", required=True, help="cand_pool.jsonl from retrieval")
    p.add_argument("--embeddings", required=True, help="item_embs.npz from retrieval")
    p.add_argument("--tracks", required=True, help="tracks.json catalog")
    p.add_argument("--out", required=True, help="learned_i2i.jsonl output")
    p.add_argument("--topk", type=int, default=200)
    p.add_argument("--seed", type=int, default=31312)
    return p.parse_args()

def discover_log_files(log_dir):
    paths = sorted(glob.glob(str(Path(log_dir) / "**" / "data.json*"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no data.json* under {log_dir}")
    return paths


def read_events(log_paths):
    rows = []
    for path in log_paths:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if rec.get("message") not in ("next", "last"):
                    continue
                rows.append((
                    int(rec["user"]),
                    int(rec["timestamp"]),
                    int(rec["track"]),
                    float(rec.get("time", 0.0) or 0.0),
                ))
    return rows


def read_candidates(path):
    pool = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            d = json.loads(raw)
            pool[int(d["item_id"])] = [int(x) for x in d["recommendations"]]
    return pool


def read_embeddings(path):
    npz = np.load(path)
    items = npz["item_ids"].astype(np.int64)
    vecs = npz["item_embs"].astype(np.float32)
    idx = {int(t): i for i, t in enumerate(items.tolist())}
    return idx, vecs


def read_artists(tracks_path):
    out = {}
    with open(tracks_path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            r = json.loads(raw)
            tid = r.get("track")
            aid = r.get("artist_id")
            if tid is None or aid is None:
                continue
            out[int(tid)] = int(aid)
    return out

def build_transitions(events):
    by_user = defaultdict(list)
    for u, ts, t, x in events:
        by_user[u].append((ts, t, x))

    transitions = []
    for plays in by_user.values():
        plays.sort()
        for i in range(len(plays) - 1):
            _, anchor, anchor_time = plays[i]
            _, cand, cand_time = plays[i + 1]
            if anchor_time < ANCHOR_MIN_TIME:
                continue
            transitions.append((anchor, cand, cand_time))
    return transitions


def aggregate_pair_stats(transitions):
    pair_plays = Counter()
    pair_completes = Counter()
    pair_skips = Counter()
    anchor_count = Counter()
    cand_count = Counter()
    cand_skips = Counter()

    for a, c, t in transitions:
        pair_plays[(a, c)] += 1
        anchor_count[a] += 1
        cand_count[c] += 1
        if t >= SKIP_THRESHOLD:
            pair_completes[(a, c)] += 1
        else:
            pair_skips[(a, c)] += 1
            cand_skips[c] += 1
    return {
        "pair_plays": pair_plays,
        "pair_completes": pair_completes,
        "pair_skips": pair_skips,
        "anchor_count": anchor_count,
        "cand_count": cand_count,
        "cand_skips": cand_skips,
        "n": sum(pair_plays.values()),
    }


def fit_beta_prior(stats):
    plays = []
    completes = []
    for (a, c), n in stats["pair_plays"].items():
        plays.append(n)
        completes.append(stats["pair_completes"].get((a, c), 0))

    plays_a = np.asarray(plays, dtype=np.float64)
    completes_a = np.asarray(completes, dtype=np.float64)
    if plays_a.sum() == 0:
        return 1.0, 1.0

    p = completes_a / np.maximum(plays_a, 1.0)
    w = plays_a / plays_a.sum()
    mean = float((w * p).sum())
    var = float((w * (p - mean) ** 2).sum())
    if mean <= 0 or mean >= 1 or var <= 0:
        return 1.0, 1.0
    factor = mean * (1.0 - mean) / var - 1.0
    if factor <= 0:
        return 1.0, 1.0
    alpha = max(0.5, mean * factor)
    beta = max(0.5, (1.0 - mean) * factor)
    return alpha, beta

def compute_pair_features(anchor, cand, rank, stats, alpha, beta,
                          embs, idx, artist_of, n_total):
    n_ac = stats["pair_plays"].get((anchor, cand), 0)
    n_a = stats["anchor_count"].get(anchor, 0)
    n_c = stats["cand_count"].get(cand, 0)
    if n_ac > 0 and n_a > 0 and n_c > 0:
        pmi = math.log((n_ac + 1.0) * n_total / ((n_a + 1.0) * (n_c + 1.0)))
    else:
        pmi = 0.0

    n_complete = stats["pair_completes"].get((anchor, cand), 0)
    bb = (alpha + n_complete) / (alpha + beta + n_ac)

    a_idx = idx.get(anchor)
    c_idx = idx.get(cand)
    if a_idx is not None and c_idx is not None:
        dot = float(np.dot(embs[a_idx], embs[c_idx]))
    else:
        dot = 0.0

    same_artist = 1.0 if artist_of.get(anchor, -1) == artist_of.get(cand, -2) else 0.0

    cand_plays = stats["cand_count"].get(cand, 0)
    cand_skip = stats["cand_skips"].get(cand, 0)
    pop_score = math.log1p(cand_plays)
    cand_skip_rate = (cand_skip + 1.0) / (cand_plays + 2.0)

    return [pmi, bb, dot, same_artist, pop_score, cand_skip_rate]


def make_training_frame(transitions, candidates, stats, alpha, beta, embs, idx, artist_of, sample_cap, seed):
    rank_lookup = {}
    for anchor, recs in candidates.items():
        for r, c in enumerate(recs):
            rank_lookup[(anchor, c)] = r

    rows, durations, events = [], [], []
    for anchor, cand, cand_time in transitions:
        rank = rank_lookup.get((anchor, cand))
        if rank is None:
            continue
        feats = compute_pair_features(
            anchor, cand, rank, stats, alpha, beta, embs, idx, artist_of, stats["n"],
        )
        rows.append(feats)
        durations.append(max(float(cand_time), EPS_DURATION))
        events.append(1 if cand_time < SKIP_THRESHOLD else 0)

    if not rows:
        raise RuntimeError("no overlap")

    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    df["duration"] = durations
    df["event"] = events

    if len(df) > sample_cap:
        rng = np.random.default_rng(seed)
        n_event = int(df["event"].sum())
        n_keep_event = min(n_event, sample_cap // 2)
        n_keep_cens = sample_cap - n_keep_event
        ev_idx = df.index[df["event"] == 1].to_numpy()
        cn_idx = df.index[df["event"] == 0].to_numpy()
        keep_ev = rng.choice(ev_idx, n_keep_event, replace=False) if len(ev_idx) > n_keep_event else ev_idx
        keep_cn = rng.choice(cn_idx, min(n_keep_cens, len(cn_idx)), replace=False) if len(cn_idx) > n_keep_cens else cn_idx
        df = df.loc[np.concatenate([keep_ev, keep_cn])].reset_index(drop=True)

    df = df.sort_values(["duration", "event"], kind="mergesort").reset_index(drop=True)
    return df


def fit_cox(df, seed):
    feats = df[FEATURE_COLS].astype(np.float64)
    means = feats.mean(axis=0)
    stds = feats.std(axis=0).replace(0.0, 1.0)
    feats_z = (feats - means) / stds

    fit_df = pd.concat([feats_z, df[["duration", "event"]]], axis=1)
    cox = CoxPHFitter(penalizer=COX_PENALIZER, l1_ratio=0.0)
    cox.fit(fit_df, duration_col="duration", event_col="event", show_progress=False)
    print(cox.summary[["coef", "exp(coef)", "p"]].to_string())
    return cox, means, stds

def score_anchor(cox, means, stds, feats):
    feats_z = (feats - means.values) / stds.values
    df = pd.DataFrame(feats_z, columns=FEATURE_COLS)
    hazard = cox.predict_partial_hazard(df).to_numpy()
    return -np.log(np.maximum(hazard, 1e-12))


def gumbel_topk(scores, anchor_id, topk, head_size, tau):
    n = scores.shape[0]
    if n <= head_size or tau <= 0.0:
        order = np.argsort(-scores)
        return order[:topk]

    head_idx = np.argpartition(-scores, head_size)[:head_size]
    head_idx = head_idx[np.argsort(-scores[head_idx])]
    rest_mask = np.ones(n, dtype=bool)
    rest_mask[head_idx] = False
    rest_idx = np.where(rest_mask)[0]

    rng = np.random.RandomState(anchor_id & 0x7fffffff)
    g = rng.gumbel(loc=0.0, scale=1.0, size=rest_idx.shape[0]).astype(np.float32)
    perturbed = scores[rest_idx] + tau * g
    tail_order = rest_idx[np.argsort(-perturbed)]
    out = np.concatenate([head_idx, tail_order])
    return out[:topk]


def rerank(candidates, cox, means, stds, stats, alpha, beta,
           embs, idx, artist_of, topk, n_total):
    out = {}
    anchor_keys = sorted(candidates.keys())
    for k, anchor in enumerate(anchor_keys):
        if k % 2000 == 0:
            print(f"  reranking {k}/{len(anchor_keys)}")

        recs = candidates[anchor]
        feats = np.empty((len(recs), len(FEATURE_COLS)), dtype=np.float64)
        for r, c in enumerate(recs):
            feats[r] = compute_pair_features(
                anchor, c, r, stats, alpha, beta, embs, idx, artist_of, n_total,
            )

        scores = score_anchor(cox, means, stds, feats)
        order = gumbel_topk(scores, anchor, topk, DETERMINISTIC_HEAD, GUMBEL_TAU)
        out[anchor] = [int(recs[i]) for i in order]
    return out


def write_jsonl(records, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for anchor, recs in records.items():
            fh.write(json.dumps({"item_id": int(anchor), "recommendations": recs}) + "\n")


def main():
    args = cli()
    np.random.seed(args.seed)

    print(f"[survival] reading logs from {args.logs}")
    log_paths = discover_log_files(args.logs)
    events = read_events(log_paths)
    print(f"  events: {len(events)}")

    transitions = build_transitions(events)
    print(f"  transitions (anchor.time>={ANCHOR_MIN_TIME}): {len(transitions)}")
    if not transitions:
        print("  no transitions — aborting", file=sys.stderr)
        sys.exit(1)

    candidates = read_candidates(args.candidates)
    idx, embs = read_embeddings(args.embeddings)
    artist_of = read_artists(args.tracks)
    print(f"  candidates: {len(candidates)} anchors, embs: {embs.shape}")

    stats = aggregate_pair_stats(transitions)
    alpha, beta = fit_beta_prior(stats)
    print(f"  beta prior (empirical-Bayes): alpha={alpha:.3f} beta={beta:.3f}")
    print(f"  global completion rate: {1.0 - sum(stats['cand_skips'].values()) / max(stats['n'], 1):.3f}")

    df = make_training_frame(
        transitions, candidates, stats, alpha, beta,
        embs, idx, artist_of, TRAIN_SAMPLE_CAP, args.seed,
    )
    print(f"  training rows: {len(df)} (events: {int(df['event'].sum())})")

    print("[survival] fitting Cox PH")
    cox, means, stds = fit_cox(df, args.seed)

    print(f"[survival] reranking top-{args.topk}")
    reranked = rerank(
        candidates, cox, means, stds, stats, alpha, beta,
        embs, idx, artist_of, args.topk, stats["n"],
    )

    write_jsonl(reranked, args.out)
    print(f"  wrote {len(reranked)} anchors → {args.out}")


if __name__ == "__main__":
    main()
