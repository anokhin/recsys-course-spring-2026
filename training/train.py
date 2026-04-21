"""
Обучает ALS на собранных sessions.jsonl и выгружает per-user top-K рекомендации
в формате, аналогичном hstu_recommendations.json.

Задача рекомендера: максимизировать mean_time_per_session. В симуляторе время
прослушки растёт с дот-продуктом эмбеддингов трека и сессионного интереса
пользователя, и штрафуется за повторное использование артиста в сессии.
Отсюда:
  - обучаем implicit.als на (user, track) матрице с весом w=log1p(time*100)
    — чем дольше слушал, тем сильнее сигнал;
  - строим список top-K по user_factors @ item_factors.T;
  - в онлайне в ридер-рекомендере будем применять динамическую дисконтку
    артиста по текущей сессии.

Сохраняем так же матрицу item factors и список артистов, чтобы рекомендер
мог пересортировать с учётом диверсификации.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

if not hasattr(np, "in1d"):
    np.in1d = np.isin  # compat для implicit + NumPy>=2


HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def load_sessions(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            if j.get("is_first"):
                continue
            rows.append((int(j["user"]), int(j["track"]), float(j["time"])))
    df = pd.DataFrame(rows, columns=["user", "track", "time"])
    df = df[df["time"] > 0].copy()
    df = df.groupby(["user", "track"], as_index=False)["time"].sum()
    return df


def build_matrix(df: pd.DataFrame, n_users: int, n_tracks: int) -> sp.csr_matrix:
    w = np.log1p(df["time"].to_numpy() * 100.0)
    mat = sp.csr_matrix(
        (w, (df["user"].to_numpy(), df["track"].to_numpy())),
        shape=(n_users, n_tracks),
        dtype=np.float32,
    )
    return mat


def fit_als(mat_train: sp.csr_matrix, factors: int, reg: float, iters: int, seed: int) -> AlternatingLeastSquares:
    mat_bm = bm25_weight(mat_train, K1=1.2, B=0.6).astype(np.float32)
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=reg,
        iterations=iters,
        alpha=30.0,
        random_state=seed,
        use_gpu=False,
    )
    model.fit(mat_bm, show_progress=True)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=str, default=str(HERE / "sessions.jsonl"))
    ap.add_argument("--out", type=str, default=str(REPO / "botify" / "data" / "personal_recommendations.json"))
    ap.add_argument("--factors-out", type=str, default=str(REPO / "botify" / "data" / "personal_factors.npz"))
    ap.add_argument("--factors", type=int, default=128)
    ap.add_argument("--reg", type=float, default=0.04)
    ap.add_argument("--iters", type=int, default=24)
    ap.add_argument("--topk", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading sessions...")
    df = load_sessions(Path(args.sessions))
    n_users = int(df["user"].max()) + 1
    n_tracks = int(df["track"].max()) + 1
    # Выравниваем размеры по известному каталогу (на всякий)
    tracks_path = REPO / "botify" / "data" / "tracks.json"
    with open(tracks_path) as f:
        max_track = max(int(json.loads(l)["track"]) for l in f)
    n_tracks = max(n_tracks, max_track + 1)
    users_path = REPO / "sim" / "data" / "users.json"
    with open(users_path) as f:
        max_user = max(int(json.loads(l)["user"]) for l in f)
    n_users = max(n_users, max_user + 1)
    print(f"  interactions={len(df):,}  n_users={n_users}  n_tracks={n_tracks}")

    mat = build_matrix(df, n_users, n_tracks)
    print(f"  sparsity={mat.nnz / (n_users * n_tracks):.6f}")

    print("Fitting ALS...")
    model = fit_als(mat, args.factors, args.reg, args.iters, args.seed)

    print("Exporting ALS item factors...")
    item_factors = model.item_factors.astype(np.float32)
    user_factors = model.user_factors.astype(np.float32)
    # Нормализуем item-факторы: score = user_vec · f_i трактуем как cosine-подобие.
    item_norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    item_factors_n = item_factors / np.clip(item_norms, 1e-8, None)

    out_factors_path = Path(args.factors_out)
    out_factors_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_factors_path,
        item_factors=item_factors_n.astype(np.float32),
        user_factors=user_factors.astype(np.float32),
    )
    print(f"Wrote {out_factors_path} (item={item_factors.shape}, user={user_factors.shape})")

    print("Exporting per-user top-K fallback list...")
    topk = args.topk
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        batch = 256
        for u0 in range(0, n_users, batch):
            u1 = min(u0 + batch, n_users)
            scores = user_factors[u0:u1] @ item_factors.T  # (b, n_tracks)
            idx = np.argpartition(-scores, topk, axis=1)[:, :topk]
            rows = np.arange(idx.shape[0])[:, None]
            top_scores = scores[rows, idx]
            order = np.argsort(-top_scores, axis=1)
            idx = idx[rows, order]
            for i, u in enumerate(range(u0, u1)):
                fout.write(json.dumps({"user": int(u), "tracks": [int(t) for t in idx[i]]}) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
