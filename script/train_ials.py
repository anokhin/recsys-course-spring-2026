"""
Train iALS on simulation logs and generate per-user top-100 recommendations.
Output format matches hstu_recommendations.json: {"user": N, "tracks": [...]}

Usage:
    python script/train_ials.py --data ./data --output botify/data/hstu_recommendations.json
"""
import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse


def ensure_implicit():
    try:
        import implicit
    except ImportError:
        print("Installing implicit library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "implicit", "-q"])


def read_logs(data_dir: Path) -> pd.DataFrame:
    paths = (
        glob.glob(str(data_dir / "*/data.json")) or
        glob.glob(str(data_dir / "**/data.json"), recursive=True)
    )
    if not paths:
        raise FileNotFoundError(f"No data.json found in {data_dir}")
    df = pd.concat([pd.read_json(p, lines=True) for p in sorted(paths)])
    print(f"Loaded {len(df):,} log rows from {len(paths)} files")
    return df


def build_interaction_matrix(df: pd.DataFrame):
    # Use only 'next' events (not 'last') to avoid double-counting
    df = df[df["message"] == "next"].copy()
    df = df[df["time"] > 0].copy()

    print(f"Interactions after filtering: {len(df):,}")

    # Aggregate time per (user, track)
    agg = df.groupby(["user", "track"])["time"].sum().reset_index()
    agg.columns = ["user", "track", "total_time"]

    # Map to consecutive indices
    users = sorted(agg["user"].unique())
    tracks = sorted(agg["track"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    track2idx = {t: i for i, t in enumerate(tracks)}

    rows = agg["user"].map(user2idx).values
    cols = agg["track"].map(track2idx).values
    data = agg["total_time"].values.astype(np.float32)

    # Normalize confidence: log-scale listened time
    data = np.log1p(data)

    mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(tracks)))
    print(f"Matrix: {len(users)} users x {len(tracks)} tracks, {mat.nnz:,} interactions")
    return mat, users, tracks


def train_ials(mat, factors=64, iterations=20, regularization=0.01):
    import implicit
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=42,
    )
    # implicit expects item x user matrix
    model.fit(mat.T.tocsr(), show_progress=True)
    return model


def generate_recommendations(model, mat, users, tracks, top_k=100):
    # Directly use factor matrices to avoid implicit's batch-recommend threading bug
    # We trained on mat.T (item x user), so implicit's naming is swapped:
    #   model.user_factors -> shape (n_items, factors)  [items were "users" in training]
    #   model.item_factors -> shape (n_users, factors)  [users were "items" in training]
    user_factors = model.item_factors  # shape: (n_users, factors)
    item_factors = model.user_factors  # shape: (n_items, factors)

    # Compute all user-item scores at once: (n_users, n_items)
    scores_matrix = user_factors @ item_factors.T

    # Zero out already-listened items
    liked = mat.tocsr()
    liked_bool = (liked > 0).toarray()
    scores_matrix[liked_bool] = -np.inf

    # Top-k item indices per user
    top_indices = np.argpartition(scores_matrix, -top_k, axis=1)[:, -top_k:]
    # Sort each user's top-k by score descending
    recs = {}
    for i, user_id in enumerate(users):
        row_scores = scores_matrix[i, top_indices[i]]
        sorted_local = np.argsort(row_scores)[::-1]
        sorted_item_idx = top_indices[i][sorted_local]
        recs[user_id] = [int(tracks[idx]) for idx in sorted_item_idx]
    return recs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Directory with simulation logs")
    parser.add_argument("--output", required=True, help="Output jsonl file path")
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    ensure_implicit()

    df = read_logs(Path(args.data))
    mat, users, tracks = build_interaction_matrix(df)
    model = train_ials(mat, factors=args.factors, iterations=args.iterations)
    recs = generate_recommendations(model, mat, users, tracks, top_k=args.top_k)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for user_id, track_list in recs.items():
            f.write(json.dumps({"user": int(user_id), "tracks": track_list}) + "\n")

    print(f"\nSaved {len(recs)} user recommendation lists to {out_path}")
    print("Next: rebuild Docker container with new recommendations")


if __name__ == "__main__":
    main()
