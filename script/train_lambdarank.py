import argparse
import glob
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs",       required=True)
    p.add_argument("--tracks",     required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--meta",       required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--topk",       type=int, default=200)
    p.add_argument("--min-listen", type=float, default=0.5)
    p.add_argument("--seed",       type=int, default=31312)
    return p.parse_args()



def read_logs(log_dir: Path) -> pd.DataFrame:
    paths = glob.glob(str(log_dir / "**/data.json*"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No logs in {log_dir}")
    return pd.concat([pd.read_json(p, lines=True) for p in sorted(paths)], ignore_index=True)


def _parse_year(v):
    if not v:
        return 0
    try:
        return int(str(v)[:4])
    except Exception:
        return 0


def read_tracks(path: Path):
    meta = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tid = int(r["track"])
            meta[tid] = {
                "artist_id":   int(r["artist_id"]) if r.get("artist_id") is not None else -1,
                "genres":      frozenset(r.get("genres") or []),
                "mood":        r.get("mood") or "",
                "year":        _parse_year(r.get("year")),
                "artist_fans": float(r.get("artist_fans") or 0),
            }
    return meta


def read_candidates(path: Path):
    cands = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            item = d.get("item_id")
            recs = d.get("recommendations")
            if item is not None and recs is not None:
                cands[int(item)] = [int(x) for x in recs]
    return cands



def build_transitions(df: pd.DataFrame, min_listen: float):
    events = df[df["message"].isin(["next", "last"])].sort_values(["user", "timestamp"])
    positives  = defaultdict(Counter)
    hard_negs  = defaultdict(Counter)
    cand_counts = Counter()

    for _, grp in events.groupby("user"):
        rows = list(grp.itertuples())
        for i in range(len(rows) - 1):
            cur, nxt = rows[i], rows[i + 1]
            anchor = int(cur.track)
            target = int(nxt.track)
            anchor_time = float(cur.time)
            target_time = float(nxt.time)
            cand_counts[target] += 1

            if anchor_time >= min_listen:
                if target_time >= min_listen:
                    positives[anchor][target]  += 1
                else:
                    hard_negs[anchor][target]  += 1

    return positives, hard_negs, cand_counts



def compute_dot_scores(anchor, candidates, embeddings, item_idx):
    a_idx = item_idx.get(anchor)
    if a_idx is None:
        return {c: 0.0 for c in candidates}
    a_emb = embeddings[a_idx]
    scores = {}
    for c in candidates:
        c_idx = item_idx.get(c)
        if c_idx is not None:
            scores[c] = float(np.dot(a_emb, embeddings[c_idx]))
        else:
            scores[c] = 0.0
    return scores


def make_features(anchor, cand, rank_in_list, artist_same_so_far,
                  dot_score, positives, hard_negs, cand_counts, track_meta):
    am = track_meta.get(anchor, {})
    cm = track_meta.get(cand,   {})

    same_artist  = 1.0 if am.get("artist_id", -1) == cm.get("artist_id", -2) else 0.0
    a_genres     = am.get("genres", frozenset())
    c_genres     = cm.get("genres", frozenset())
    genre_overlap = len(a_genres & c_genres) / max(1, math.sqrt(len(a_genres) * len(c_genres) + 1e-6))
    same_mood    = 1.0 if am.get("mood") and am.get("mood") == cm.get("mood") else 0.0
    a_yr, c_yr   = am.get("year", 0), cm.get("year", 0)
    year_diff    = abs(a_yr - c_yr) / 50.0 if a_yr and c_yr else 0.0
    log_fans     = math.log1p(cm.get("artist_fans", 0))

    log_cand_cnt = math.log1p(cand_counts.get(cand, 0))
    log_trans    = math.log1p(positives.get(anchor, {}).get(cand, 0))
    log_skip     = math.log1p(hard_negs.get(anchor, {}).get(cand, 0))

    diversity_pen = float(artist_same_so_far) * same_artist
    rank_score = 1.0 / (rank_in_list + 1)

    return [
        dot_score,
        same_artist,
        genre_overlap,
        same_mood,
        year_diff,
        log_fans,
        log_cand_cnt,
        log_trans,
        log_skip,
        diversity_pen,
        rank_score,
    ]


FEATURE_NAMES = [
    "dot_score", "same_artist", "genre_overlap", "same_mood", "year_diff",
    "log_fans", "log_cand_cnt", "log_trans", "log_skip", "diversity_pen", "rank_score",
]


def relevance_label(anchor, cand, positives, hard_negs, min_listen):
    if positives.get(anchor, {}).get(cand, 0) > 0:
        return 2
    if hard_negs.get(anchor, {}).get(cand, 0) > 0:
        return 1
    return 0



def build_train_data(candidates, positives, hard_negs, cand_counts,
                     track_meta, embeddings, item_idx):
    X, y, groups = [], [], []

    for anchor, cand_list in candidates.items():
        dot_scores = compute_dot_scores(anchor, cand_list, embeddings, item_idx)

        artist_seen = Counter()
        for rank, cand in enumerate(cand_list):
            am = track_meta.get(anchor, {})
            cm = track_meta.get(cand,   {})
            a_artist = cm.get("artist_id", -1)
            same_so_far = artist_seen[a_artist]

            feat = make_features(
                anchor, cand, rank, same_so_far,
                dot_scores[cand], positives, hard_negs, cand_counts, track_meta,
            )
            label = relevance_label(anchor, cand, positives, hard_negs, min_listen=0.5)

            X.append(feat)
            y.append(label)
            if a_artist >= 0:
                artist_seen[a_artist] += int(cm.get("artist_id", -1) == am.get("artist_id", -2))

        groups.append(len(cand_list))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), groups



def train_lambdarank(X, y, groups, seed):
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[5, 10, 20],
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X, y, group=groups)
    print(f"  LambdaRank trained on {len(y)} (anchor, candidate) pairs")
    return model


def rerank(candidates, model, positives, hard_negs, cand_counts,
           track_meta, embeddings, item_idx, topk):
    results = {}
    for i, (anchor, cand_list) in enumerate(candidates.items()):
        if i % 2000 == 0:
            print(f"  reranking {i}/{len(candidates)}")

        dot_scores = compute_dot_scores(anchor, cand_list, embeddings, item_idx)
        artist_seen = Counter()
        X_pred = []

        for rank, cand in enumerate(cand_list):
            cm = track_meta.get(cand, {})
            a_artist = cm.get("artist_id", -1)
            same_so_far = artist_seen[a_artist]

            feat = make_features(
                anchor, cand, rank, same_so_far,
                dot_scores[cand], positives, hard_negs, cand_counts, track_meta,
            )
            X_pred.append(feat)

            if a_artist >= 0:
                am = track_meta.get(anchor, {})
                artist_seen[a_artist] += int(a_artist == am.get("artist_id", -2))

        scores = model.predict(np.array(X_pred, dtype=np.float32))
        order  = np.argsort(-scores)
        results[anchor] = [cand_list[j] for j in order[:topk]]

    return results



def main():
    args = parse_args()

    print("Loading data")
    df         = read_logs(Path(args.logs))
    track_meta = read_tracks(Path(args.tracks))
    candidates = read_candidates(Path(args.candidates))

    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    embeddings = meta["embeddings"]
    item_idx   = meta["item_idx"]

    print("Building transition stats")
    positives, hard_negs, cand_counts = build_transitions(df, min_listen=args.min_listen)
    print(f"  anchors with positives: {len(positives)}, with hard_negs: {len(hard_negs)}")

    print("Building training data")
    X, y, groups = build_train_data(
        candidates, positives, hard_negs, cand_counts,
        track_meta, embeddings, item_idx,
    )
    print(f"  rows={len(y)}, label_dist={dict(zip(*np.unique(y, return_counts=True)))}")

    print("Training LambdaRank")
    ranker = train_lambdarank(X, y, groups, seed=args.seed)

    print("Reranking candidates")
    reranked = rerank(
        candidates, ranker, positives, hard_negs, cand_counts,
        track_meta, embeddings, item_idx, topk=args.topk,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for anchor, recs in reranked.items():
            f.write(json.dumps({"item_id": int(anchor), "recommendations": [int(r) for r in recs]}) + "\n")
    print(f"wrote {len(reranked)} anchors -> {out}")


if __name__ == "__main__":
    main()
