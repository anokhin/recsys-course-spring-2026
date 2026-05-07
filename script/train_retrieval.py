import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp


os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


MIN_LISTEN = 0.5
MAX_ANCHORS = 15000
ALS_FACTORS = 64
ALS_ITERATIONS = 25
ALS_REGULARIZATION = 0.05
ALS_ALPHA = 40.0


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", required=True, help="Directory with collected botify logs")
    p.add_argument("--tracks", required=True, help="Path to tracks.json catalog (unused; kept for makefile parity)")
    p.add_argument("--cand-out", required=True, help="JSONL with (item_id, recommendations)")
    p.add_argument("--emb-out", required=True, help=".npz with item_ids and item_embs")
    p.add_argument("--topk", type=int, default=500)
    p.add_argument("--factors", type=int, default=ALS_FACTORS)
    p.add_argument("--iterations", type=int, default=ALS_ITERATIONS)
    p.add_argument("--seed", type=int, default=31312)
    return p.parse_args()


def discover_log_files(log_dir):
    paths = sorted(glob.glob(str(Path(log_dir) / "**" / "data.json*"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no data.json* under {log_dir}")
    return paths


def stream_interactions(log_paths, min_listen):
    for path in log_paths:
        with open(path, "r", encoding="utf-8") as fh:
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
                t = rec.get("time", 0.0)
                if t is None or t < min_listen:
                    continue
                yield int(rec["user"]), int(rec["track"]), float(t)


def build_user_item_matrix(interactions, alpha):
    users = sorted({u for u, _, _ in interactions})
    items = sorted({t for _, t, _ in interactions})
    user_idx = {u: i for i, u in enumerate(users)}
    item_idx = {t: i for i, t in enumerate(items)}

    rows, cols, vals = [], [], []
    for u, t, listen_time in interactions:
        rows.append(user_idx[u])
        cols.append(item_idx[t])
        vals.append(1.0 + alpha * float(np.log1p(listen_time)))

    mat = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(users), len(items)),
        dtype=np.float32,
    )
    mat.sum_duplicates()
    return mat, users, items


def fit_als(matrix, factors, iterations, regularization, seed):
    from implicit.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        use_gpu=False,
        random_state=seed,
    )
    model.fit(matrix, show_progress=False)
    return model


def pick_anchors(matrix, items, max_anchors):
    counts = np.asarray(matrix.getnnz(axis=0)).ravel()
    order = np.argsort(-counts)
    return order[: min(max_anchors, len(order))]


def candidate_pool(item_factors, anchor_idx, topk):
    n_items, dim = item_factors.shape
    pool = np.empty((len(anchor_idx), topk), dtype=np.int32)

    chunk = 256
    for start in range(0, len(anchor_idx), chunk):
        end = min(start + chunk, len(anchor_idx))
        anchors = anchor_idx[start:end]
        scores = item_factors[anchors] @ item_factors.T   # (B, n_items)

        for j, a in enumerate(anchors):
            scores[j, a] = -np.inf

        head = np.argpartition(-scores, topk, axis=1)[:, :topk]
        for j in range(end - start):
            row_scores = scores[j, head[j]]
            order = np.argsort(-row_scores)
            pool[start + j] = head[j, order]
    return pool


def write_jsonl(items, anchor_idx, pool, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    items_arr = np.asarray(items, dtype=np.int64)
    with p.open("w", encoding="utf-8") as fh:
        for j, anchor_pos in enumerate(anchor_idx):
            recs = items_arr[pool[j]].tolist()
            fh.write(json.dumps({
                "item_id": int(items_arr[anchor_pos]),
                "recommendations": [int(x) for x in recs],
            }) + "\n")


def main():
    args = cli()
    np.random.seed(args.seed)

    print(f"[retrieval] discovering logs under {args.logs}")
    log_paths = discover_log_files(args.logs)
    print(f"  found {len(log_paths)} log file(s)")

    interactions = list(stream_interactions(log_paths, MIN_LISTEN))
    if not interactions:
        print("  no interactions after filtering — aborting", file=sys.stderr)
        sys.exit(1)
    print(f"  positive interactions (time>={MIN_LISTEN}): {len(interactions)}")

    matrix, users, items = build_user_item_matrix(interactions, ALS_ALPHA)
    print(f"  matrix: users={matrix.shape[0]} items={matrix.shape[1]} nnz={matrix.nnz}")

    print(f"[retrieval] fit ALS (factors={args.factors}, iterations={args.iterations})")
    model = fit_als(matrix, args.factors, args.iterations, ALS_REGULARIZATION, args.seed)
    item_factors = np.asarray(model.item_factors, dtype=np.float32)
    print(f"  item factors: {item_factors.shape}")

    anchor_idx = pick_anchors(matrix, items, MAX_ANCHORS)
    print(f"[retrieval] computing top-{args.topk} for {len(anchor_idx)} anchors")
    pool = candidate_pool(item_factors, anchor_idx, args.topk)

    write_jsonl(items, anchor_idx, pool, args.cand_out)
    print(f"  wrote {len(anchor_idx)} anchors → {args.cand_out}")

    emb_out = Path(args.emb_out)
    emb_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        emb_out,
        item_ids=np.asarray(items, dtype=np.int64),
        item_embs=item_factors,
    )
    print(f"  wrote item embeddings → {emb_out}")


if __name__ == "__main__":
    main()
