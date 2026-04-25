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


def build_transition_i2i(
    raw_events: pd.DataFrame,
    n_tracks: int,
    top_k: int = 10,
    smoothing_with_collab: np.ndarray | None = None,
    smoothing_alpha: float = 0.0,
) -> Dict[int, List[int]]:
    """For each track a, top-K tracks that frequently followed a in user sessions
    weighted by the listened-fraction (listen_time) of the next track.

    A "session" is a consecutive run of events for a single user that ends at a
    'last' message. We accumulate transitions only within a session.

    If `smoothing_with_collab` is provided (n_tracks, n_tracks) cosine-sim matrix
    or (n_tracks, k) item factor matrix, we add `smoothing_alpha * collab_score`
    on top of transition counts for ranking. This helps fill in cold pairs where
    no real transition was observed.
    """
    df = raw_events.sort_values(["user", "timestamp"]).reset_index(drop=True)

    transitions = sp.lil_matrix((n_tracks, n_tracks), dtype=np.float32)
    prev_user = None
    prev_track = None
    for user, track, listen_time, msg in df[
        ["user", "track", "listen_time", "message"]
    ].itertuples(index=False):
        if user == prev_user and prev_track is not None:
            transitions[int(prev_track), int(track)] += float(listen_time)
        if msg == "last":
            prev_user = None
            prev_track = None
        else:
            prev_user = user
            prev_track = int(track)

    transitions = transitions.tocsr()

    use_smoothing = (
        smoothing_with_collab is not None and smoothing_alpha > 0.0
    )
    if use_smoothing:
        f = smoothing_with_collab
        f_norms = np.linalg.norm(f, axis=1, keepdims=True)
        f_norms[f_norms == 0] = 1.0
        f_n = (f / f_norms).astype(np.float32)
    else:
        f_n = None

    out: Dict[int, List[int]] = {}
    for a in range(n_tracks):
        row = transitions.getrow(a)
        scores = row.toarray().ravel().astype(np.float32)
        if use_smoothing:
            scores = scores + smoothing_alpha * (f_n[a:a + 1] @ f_n.T).ravel()
        scores[a] = -np.inf
        if scores.max() <= -np.inf + 1:
            continue
        top_idx = np.argpartition(-scores, top_k)[:top_k]
        ranked = top_idx[np.argsort(-scores[top_idx])]
        out[a] = ranked.astype(int).tolist()
    return out


def build_item_item(
    item_factors: np.ndarray,
    content_emb: np.ndarray,
    top_k: int = 10,
    alpha: float = 0.5,
    batch_size: int = 512,
) -> Dict[int, List[int]]:
    """Item-item top-K combining iALS item factors (collab) and content embeddings.

    score(i, j) = alpha * cosine(iALS_factors[i], iALS_factors[j])
                + (1 - alpha) * cosine(content_emb[i], content_emb[j])

    Both vectors are L2-normalized so dot product == cosine similarity.
    Items with all-zero iALS factors (no interactions) fall back to pure content sim.
    Self is masked out before top-K selection.
    """
    n_items = item_factors.shape[0]
    assert content_emb.shape[0] == n_items

    f_norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    f_norms[f_norms == 0] = 1.0
    item_factors_n = (item_factors / f_norms).astype(np.float32)
    content_n = content_emb.astype(np.float32)  # already normalized at embed time

    out: Dict[int, List[int]] = {}
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        sim_collab = item_factors_n[start:end] @ item_factors_n.T
        sim_content = content_n[start:end] @ content_n.T
        sim = alpha * sim_collab + (1.0 - alpha) * sim_content

        for local_idx, global_idx in enumerate(range(start, end)):
            sim[local_idx, global_idx] = -np.inf

        # argpartition for unsorted top-K, then sort that small slice
        top_unsorted = np.argpartition(-sim, top_k, axis=1)[:, :top_k]
        for local_idx, global_idx in enumerate(range(start, end)):
            cand = top_unsorted[local_idx]
            ranked = cand[np.argsort(-sim[local_idx, cand])]
            out[int(global_idx)] = ranked.astype(int).tolist()
    return out


def train_and_recommend(
    logs: pd.DataFrame,
    track_embeddings: np.ndarray,
    all_user_ids: Iterable[int],
    out_path: Path,
    i2i_out_path: Path | None = None,
    i2i_mode: str = "blend",
    raw_events: pd.DataFrame | None = None,
    top_n: int = 10,
    ials_factors: int = 64,
    ials_iterations: int = 20,
    ials_top_k: int = 200,
    catboost_iterations: int = 500,
    catboost_depth: int = 6,
    catboost_lr: float = 0.05,
    i2i_alpha: float = 0.5,
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

    if i2i_out_path is not None:
        item_factors = np.asarray(model.item_factors)
        if item_factors.shape[0] != n_tracks:
            raise RuntimeError(
                f"item_factors shape {item_factors.shape} doesn't match n_tracks={n_tracks}"
            )

        if i2i_mode == "blend":
            i2i = build_item_item(
                item_factors=item_factors,
                content_emb=track_embeddings,
                top_k=top_n,
                alpha=i2i_alpha,
            )
        elif i2i_mode == "transition":
            assert raw_events is not None, "raw_events required for transition i2i"
            i2i = build_transition_i2i(
                raw_events=raw_events,
                n_tracks=n_tracks,
                top_k=top_n,
                smoothing_with_collab=item_factors,
                smoothing_alpha=i2i_alpha,
            )
        else:
            raise ValueError(f"unknown i2i_mode: {i2i_mode}")

        # For tracks with no entry from the chosen method, fall back to
        # blended iALS+content so every item has a top-K (covers cold
        # tracks for the transition method).
        missing = [t for t in range(n_tracks) if t not in i2i or len(i2i[t]) < top_n]
        if missing and i2i_mode == "transition":
            backfill = build_item_item(
                item_factors=item_factors,
                content_emb=track_embeddings,
                top_k=top_n,
                alpha=0.5,
            )
            for t in missing:
                i2i[t] = backfill.get(t, [])

        i2i_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(i2i_out_path, "w") as f:
            for item_id in range(n_tracks):
                f.write(json.dumps({
                    "item_id": int(item_id),
                    "recommendations": i2i.get(item_id, []),
                }) + "\n")

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
    parser.add_argument(
        "--i2i-output",
        type=Path,
        default=None,
        help="If set, also write item-item top-K JSONL to this path.",
    )
    parser.add_argument(
        "--i2i-alpha",
        type=float,
        default=0.5,
        help="In blend mode: weight on iALS vs content. "
             "In transition mode: weight on iALS-factor smoothing added to transition counts.",
    )
    parser.add_argument(
        "--i2i-mode",
        choices=["blend", "transition"],
        default="blend",
        help="blend = iALS item-factor cosine + content embedding cosine. "
             "transition = sequential transition counts from logs (+ iALS smoothing).",
    )
    parser.add_argument("--seed", type=int, default=31312)
    parser.add_argument("--experiment", default="HSTU")
    parser.add_argument(
        "--control-label",
        default=None,
        help="If set, keep only rows where experiments[<experiment>] == this label. "
             "Default: None (use all rows regardless of arm).",
    )
    args = parser.parse_args()

    from script.load_logs import load_control_arm, load_raw_events

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

    raw_events = None
    if args.i2i_mode == "transition":
        raw_events = load_raw_events(args.logs)
        print(f"loaded raw events: {len(raw_events)} rows")

    embs = np.load(args.embeddings)
    print(f"loaded embeddings: shape={embs.shape}")

    all_user_ids = sorted(logs["user"].unique().astype(int).tolist())

    train_and_recommend(
        logs=logs,
        track_embeddings=embs,
        all_user_ids=all_user_ids,
        out_path=args.output,
        i2i_out_path=args.i2i_output,
        i2i_mode=args.i2i_mode,
        raw_events=raw_events,
        i2i_alpha=args.i2i_alpha,
        seed=args.seed,
    )
    print(f"wrote recommendations: {args.output}")
    if args.i2i_output is not None:
        print(f"wrote i2i recommendations: {args.i2i_output}")


if __name__ == "__main__":
    _cli()
