"""
End-to-end training: iALS retrieval + CatBoost reranker with content
features, producing botify/data/our_recommendations.jsonl.

CLI usage:
    python script/train_recsys.py \
        --logs ./data/run_train \
        --tracks botify/data/tracks.json \
        --embeddings botify/data/track_embeddings.npy \
        --output botify/data/our_recommendations.jsonl \
        --seed 31312
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _train_ials(
    logs: pd.DataFrame,
    n_tracks: int,
    factors: int,
    iterations: int,
    seed: int,
):
    import implicit

    user_ids = sorted(logs["user"].unique().tolist())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    rows = logs["user"].map(user_to_idx).to_numpy()
    cols = logs["track"].to_numpy()
    vals = logs["listen_time"].to_numpy(dtype=np.float32)

    user_item = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(user_ids), n_tracks),
        dtype=np.float32,
    )

    np.random.seed(seed)
    random.seed(seed)
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        regularization=0.01,
        iterations=iterations,
        use_gpu=False,
        random_state=seed,
    )
    model.fit(user_item, show_progress=False)
    return model, user_ids, user_item


def _ials_top_k(model, user_item, top_k: int):
    n_users = user_item.shape[0]
    cand_ids, cand_scores = model.recommend(
        userid=np.arange(n_users),
        user_items=user_item,
        N=top_k,
        filter_already_liked_items=False,
    )
    return cand_ids, cand_scores


def _user_vibe_vectors(
    logs: pd.DataFrame,
    track_embeddings: np.ndarray,
    user_ids: List[int],
) -> np.ndarray:
    dim = track_embeddings.shape[1]
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    out = np.zeros((len(user_ids), dim), dtype=np.float32)
    weights_sum = np.zeros(len(user_ids), dtype=np.float32)

    for user, track, listen_time in logs[["user", "track", "listen_time"]].itertuples(index=False):
        if user not in user_to_idx:
            continue
        i = user_to_idx[user]
        out[i] += listen_time * track_embeddings[track]
        weights_sum[i] += listen_time

    nonzero = weights_sum > 0
    out[nonzero] /= weights_sum[nonzero, None]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out /= norms
    return out


def _build_reranker_features(
    cand_ids: np.ndarray,
    cand_scores: np.ndarray,
    user_ids: List[int],
    user_vibe: np.ndarray,
    track_embeddings: np.ndarray,
    track_popularity: np.ndarray,
) -> pd.DataFrame:
    n_users, top_k = cand_ids.shape
    rows = []
    for u_idx in range(n_users):
        user = user_ids[u_idx]
        vibe = user_vibe[u_idx]
        for rank, (track, score) in enumerate(
            zip(cand_ids[u_idx], cand_scores[u_idx])
        ):
            t_emb = track_embeddings[int(track)]
            rows.append({
                "user": user,
                "track": int(track),
                "ials_score": float(score),
                "ials_rank": int(rank),
                "vibe_cos": float(np.dot(vibe, t_emb)),
                "track_pop": float(track_popularity[int(track)]),
            })
    return pd.DataFrame(rows)


def train_and_recommend(
    logs: pd.DataFrame,
    track_embeddings: np.ndarray,
    all_user_ids: Iterable[int],
    out_path: Path,
    top_n: int = 10,
    ials_factors: int = 64,
    ials_iterations: int = 20,
    ials_top_k: int = 200,
    catboost_iterations: int = 500,
    catboost_depth: int = 6,
    catboost_lr: float = 0.05,
    seed: int = 31312,
) -> None:
    n_tracks = track_embeddings.shape[0]

    track_pop = np.zeros(n_tracks, dtype=np.float32)
    if not logs.empty:
        sums = logs.groupby("track")["listen_time"].sum()
        track_pop[sums.index.to_numpy()] = sums.to_numpy(dtype=np.float32)

    pop_top_n: List[int] = (
        np.argsort(-track_pop)[:top_n].astype(int).tolist()
        if track_pop.sum() > 0
        else list(range(top_n))
    )

    if logs.empty:
        with open(out_path, "w") as f:
            for u in all_user_ids:
                f.write(
                    json.dumps({"user": int(u), "tracks": pop_top_n}) + "\n"
                )
        return

    model, ials_user_ids, user_item = _train_ials(
        logs, n_tracks, ials_factors, ials_iterations, seed
    )
    cand_ids, cand_scores = _ials_top_k(model, user_item, ials_top_k)

    user_vibe = _user_vibe_vectors(logs, track_embeddings, ials_user_ids)

    feats = _build_reranker_features(
        cand_ids, cand_scores, ials_user_ids,
        user_vibe, track_embeddings, track_pop,
    )

    label_lookup = (
        logs.groupby(["user", "track"])["listen_time"].sum().to_dict()
    )
    feats["listen_time"] = [
        float(label_lookup.get((u, t), 0.0))
        for u, t in feats[["user", "track"]].itertuples(index=False)
    ]

    feature_cols = ["ials_score", "ials_rank", "vibe_cos", "track_pop"]

    from catboost import CatBoostRegressor

    reg = CatBoostRegressor(
        iterations=catboost_iterations,
        depth=catboost_depth,
        learning_rate=catboost_lr,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
    )
    reg.fit(feats[feature_cols], feats["listen_time"])
    feats["pred"] = reg.predict(feats[feature_cols])

    feats = feats.sort_values(["user", "pred"], ascending=[True, False])

    user_to_top: Dict[int, List[int]] = {}
    for user, group in feats.groupby("user", sort=False):
        user_to_top[int(user)] = (
            group["track"].astype(int).head(top_n).tolist()
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for u in all_user_ids:
            tracks = user_to_top.get(int(u))
            if tracks is None or len(tracks) < top_n:
                fill = [t for t in pop_top_n if not tracks or t not in tracks]
                tracks = (tracks or []) + fill[: top_n - len(tracks or [])]
                tracks = tracks[:top_n]
            f.write(json.dumps({"user": int(u), "tracks": tracks}) + "\n")


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=31312)
    parser.add_argument("--experiment", default="HSTU")
    parser.add_argument(
        "--control-label",
        default=None,
        help="If set, keep only rows where experiments[<experiment>] == this label. "
             "Default: None (use all rows regardless of arm).",
    )
    args = parser.parse_args()

    from script.load_logs import load_control_arm

    logs = load_control_arm(
        args.logs,
        experiment_name=args.experiment,
        control_label=args.control_label,
        aggregate=True,
    )
    print(
        f"loaded logs: {len(logs)} (user, track) rows, "
        f"{logs['user'].nunique()} users"
    )

    embs = np.load(args.embeddings)
    print(f"loaded embeddings: shape={embs.shape}")

    all_user_ids = sorted(logs["user"].unique().astype(int).tolist())

    train_and_recommend(
        logs=logs,
        track_embeddings=embs,
        all_user_ids=all_user_ids,
        out_path=args.output,
        seed=args.seed,
    )
    print(f"wrote recommendations: {args.output}")


if __name__ == "__main__":
    _cli()
