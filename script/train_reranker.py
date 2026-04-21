"""
Train LightGBM reranker from botify data logs + train item2vec.

This module has two halves:
  1. Pure helpers (sessionize_log, build_candidate_features, extract_training_rows)
     — imported by tests and by the inference-time reranker.
  2. CLI `main()` that wires everything together (Task 7).
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


FEATURE_ORDER = [
    "history_len",
    "mean_prev_time",
    "last_time",
    "prev_artist_match",
    "candidate_rank",
    "i2v_cos_mean",
    "i2v_cos_last",
    "candidate_seen_before",
]


def sessionize_log(df) -> List[dict]:
    """Group a flat data-log dataframe into per-user sessions.

    A session starts at the first `next` event for a user and ends at the
    first following `last` event.
    """
    import pandas as pd
    df = df.sort_values(["user", "timestamp"])
    sessions: List[dict] = []
    for user, udf in df.groupby("user"):
        events: List[dict] = []
        for _, row in udf.iterrows():
            rec = row.get("recommendation")
            events.append({
                "track": int(row["track"]),
                "time": float(row["time"]),
                "recommendation": (
                    int(rec) if rec is not None and pd.notna(rec) else None
                ),
                "message": row["message"],
            })
            if row["message"] == "last":
                sessions.append({"user": int(user), "events": events})
                events = []
    return sessions


def build_candidate_features(
    history,
    candidate: int,
    candidate_rank: int,
    track_meta: Dict[int, dict],
    item2vec: Optional[Dict[int, np.ndarray]] = None,
) -> dict:
    """Features for a (session_prefix, candidate) pair. Used both at train and inference."""
    prev_times = [t for _, t in history]
    prev_tracks = [tr for tr, _ in history]
    last_track = prev_tracks[-1] if prev_tracks else None

    last_artist = track_meta.get(last_track, {}).get("artist") if last_track is not None else None
    cand_artist = track_meta.get(candidate, {}).get("artist")
    artist_match = int(last_artist is not None and last_artist == cand_artist)

    i2v_cos_mean = 0.0
    i2v_cos_last = 0.0
    if item2vec and candidate in item2vec:
        cv = item2vec[candidate]
        cv_norm = float(np.linalg.norm(cv)) + 1e-9
        sims = []
        for t in prev_tracks:
            v = item2vec.get(t)
            if v is None:
                continue
            sims.append(float(np.dot(v, cv) / (float(np.linalg.norm(v)) * cv_norm + 1e-9)))
        if sims:
            i2v_cos_mean = float(np.mean(sims))
            i2v_cos_last = sims[-1]

    return {
        "history_len": len(history),
        "mean_prev_time": float(np.mean(prev_times)) if prev_times else 0.0,
        "last_time": float(prev_times[-1]) if prev_times else 0.0,
        "prev_artist_match": artist_match,
        "candidate_rank": int(candidate_rank),
        "i2v_cos_mean": i2v_cos_mean,
        "i2v_cos_last": i2v_cos_last,
        "candidate_seen_before": int(candidate in prev_tracks),
    }


def extract_training_rows(
    session: dict,
    track_meta: Dict[int, dict],
    item2vec: Optional[Dict[int, np.ndarray]] = None,
) -> List[dict]:
    """
    For each `next` event in a session with a recorded recommendation, produce one
    training row: features of (history-so-far, recommended track) + binary label
    `listen_time > 0.5`.
    """
    rows: List[dict] = []
    history: List[tuple] = []
    user = session["user"]
    for ev in session["events"]:
        if ev["message"] == "next" and ev["recommendation"] is not None:
            candidate = int(ev["recommendation"])
            feats = build_candidate_features(
                history=list(history),
                candidate=candidate,
                candidate_rank=0,
                track_meta=track_meta,
                item2vec=item2vec,
            )
            feats["user"] = user
            feats["candidate"] = candidate
            feats["label"] = int(ev["time"] > 0.5)
            rows.append(feats)
        history.append((int(ev["track"]), float(ev["time"])))
    return rows


def _load_logs(paths: Iterable[str]):
    import pandas as pd
    dfs = []
    for p in paths:
        df = pd.read_json(p, lines=True)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _train_item2vec(sessions: List[dict], dim: int = 32, min_count: int = 2):
    from gensim.models import Word2Vec
    sentences = [[str(e["track"]) for e in s["events"]] for s in sessions]
    model = Word2Vec(
        sentences=sentences, vector_size=dim, window=5, min_count=min_count,
        workers=2, epochs=10, sg=1, seed=1234,
    )
    return {int(k): np.asarray(model.wv[k], dtype=np.float32) for k in model.wv.index_to_key}


def _load_track_meta(path: str) -> Dict[int, dict]:
    meta: Dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            meta[int(row["track"])] = {"artist": row["artist"]}
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="glob for data.json logs")
    ap.add_argument("--tracks", required=True, help="path to botify tracks.json")
    ap.add_argument("--out", required=True, help="artifact output dir")
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log_paths = sorted(glob.glob(args.logs, recursive=True))
    if not log_paths:
        raise SystemExit(f"No logs matched {args.logs}")
    print(f"Loading {len(log_paths)} log files")
    df = _load_logs(log_paths)
    df = df[df["message"].isin({"next", "last"})]

    sessions = sessionize_log(df)
    print(f"Built {len(sessions)} sessions")

    track_meta = _load_track_meta(args.tracks)
    i2v = _train_item2vec(sessions)
    print(f"Trained item2vec: {len(i2v)} tracks")

    rows: List[dict] = []
    for s in sessions:
        rows.extend(extract_training_rows(s, track_meta, i2v))
    import pandas as pd
    train_df = pd.DataFrame(rows)
    if len(train_df) < 500:
        raise SystemExit(f"Too few training rows: {len(train_df)}")
    print(f"Training rows: {len(train_df)} | positive rate: {train_df['label'].mean():.3f}")

    users = train_df["user"].unique()
    rng = np.random.default_rng(0)
    val_users = set(rng.choice(users, size=max(1, len(users) // 5), replace=False))
    val_mask = train_df["user"].isin(val_users)

    import lightgbm as lgb
    X_cols = FEATURE_ORDER
    X_train = train_df.loc[~val_mask, X_cols].values
    y_train = train_df.loc[~val_mask, "label"].values
    X_val = train_df.loc[val_mask, X_cols].values
    y_val = train_df.loc[val_mask, "label"].values

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    params = dict(
        objective="binary", learning_rate=0.05, num_leaves=31,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        min_data_in_leaf=20, metric="auc", verbose=-1, seed=1234,
        deterministic=True, force_row_wise=True,
    )
    booster = lgb.train(
        params, train_set, num_boost_round=500, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(25)],
    )
    print(f"Best val AUC: {booster.best_score['valid_0']['auc']:.4f}")

    booster.save_model(str(out / "model.txt"))
    with open(out / "item2vec.pkl", "wb") as f:
        pickle.dump(i2v, f)
    with open(out / "meta.json", "w") as f:
        json.dump({
            "feature_order": FEATURE_ORDER,
            "top_k": args.top_k,
            "num_train_rows": int(len(train_df)),
            "positive_rate": float(train_df["label"].mean()),
        }, f, indent=2)
    print(f"Artifacts written to {out}")


if __name__ == "__main__":
    main()
