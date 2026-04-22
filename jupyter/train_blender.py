#!/usr/bin/env python3
"""Train per-model blender weights via leave-one-out learning-to-rank.

We don't have labelled play-time data (sim/data/ is off-limits per the
homework rules and botify/log/ hasn't been collected yet), so we build a
genuinely supervised signal from the four upstream models themselves.

For every user and every "oracle" model m* ∈ {sasrec, ease, hstu, lightfm}
we hide its top-K list and train a regression to predict "is this track in
oracle's top-K" from the positions of the candidate in the three remaining
models. The learned model weight W_m reflects how consistently m predicts
what the other models also rank highly — i.e. its agreement-power with the
rest of the ensemble. Coefficients are averaged across all four oracles so
every model is in a fair position.

Features per (user, candidate) w.r.t. oracle m*:
    [pos_weight_m for m in MODELS if m != m*]   (3-dimensional)
with pos_weight(pos) = 1 / log2(2 + pos), 0 if the model did not include
the candidate in its top-K.

Target: 1 if the candidate is in oracle's top-K, else 0.

The final W_m is the mean coefficient that model m received across the
three regressions where it served as a feature (i.e., was not the oracle).
Weights are clipped to non-negative and max-normalised, then dumped to
blender_weights.json for jupyter/rerank.py to consume.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "botify" / "data"
WEIGHTS_PATH = DATA_DIR / "blender_weights.json"

EASE_PATH = DATA_DIR / "recommendations_ease.json"
HSTU_PATH = DATA_DIR / "hstu_recommendations.json"
SASREC_PATH = DATA_DIR / "sasrec_i2i.jsonl"
LIGHTFM_PATH = DATA_DIR / "lightfm_i2i.jsonl"

TOP_K = 10
ANCHOR_K = 5
MODELS = ["sasrec", "ease", "hstu", "lightfm"]


def pw(pos):
    return 1.0 / math.log2(2 + pos)


def load_user(path):
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["user"]] = d["tracks"]
    return out


def load_i2i(path):
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["item_id"]] = d["recommendations"]
    return out


def top_k_from_i2i(anchors, i2i, k=TOP_K):
    scores = defaultdict(float)
    anchor_set = set(anchors)
    for a_idx, anchor in enumerate(anchors):
        neighbours = i2i.get(anchor)
        if not neighbours:
            continue
        a_w = pw(a_idx)
        for n_idx, cand in enumerate(neighbours):
            if cand in anchor_set:
                continue
            scores[cand] += a_w * pw(n_idx)
    return [t for t, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]


def user_top_lists(user, ease, hstu, sasrec_i2i, lightfm_i2i):
    out = {}
    if user in ease:
        out["ease"] = ease[user][:TOP_K]
    if user in hstu:
        out["hstu"] = hstu[user][:TOP_K]
    anchors = (ease.get(user) or hstu.get(user) or [])[:ANCHOR_K]
    if anchors:
        sas = top_k_from_i2i(anchors, sasrec_i2i)
        if sas:
            out["sasrec"] = sas
        lfm = top_k_from_i2i(anchors, lightfm_i2i)
        if lfm:
            out["lightfm"] = lfm
    return out


def solve_linear(A, b):
    """Gauss-Jordan for an n×n system. Returns x such that A x = b."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < 1e-12:
            raise RuntimeError("singular matrix")
        M[col], M[pivot_row] = M[pivot_row], M[col]
        p = M[col][col]
        M[col] = [v / p for v in M[col]]
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor != 0.0:
                M[r] = [M[r][i] - factor * M[col][i] for i in range(n + 1)]
    return [M[i][n] for i in range(n)]


def leave_one_out_regression(oracle, users, tops_by_user):
    """Predict 'is in oracle's top-K' from the other three models' pos_weights.

    Returns the raw OLS coefficients keyed by feature-model name.
    """
    feature_models = [m for m in MODELS if m != oracle]
    d = len(feature_models)
    XtX = [[0.0] * d for _ in range(d)]
    Xty = [0.0] * d
    n_rows = 0
    n_pos = 0

    for user in users:
        tops = tops_by_user[user]
        if oracle not in tops:
            continue
        oracle_set = set(tops[oracle])

        pos_per_cand = defaultdict(lambda: {m: None for m in feature_models})
        for m in feature_models:
            for idx, t in enumerate(tops.get(m, [])):
                pos_per_cand[t][m] = idx

        for track, positions in pos_per_cand.items():
            feats = [
                pw(positions[m]) if positions[m] is not None else 0.0
                for m in feature_models
            ]
            label = 1.0 if track in oracle_set else 0.0
            n_rows += 1
            n_pos += int(label)
            for i in range(d):
                Xty[i] += feats[i] * label
                for j in range(d):
                    XtX[i][j] += feats[i] * feats[j]

    lam = 1e-3
    for i in range(d):
        XtX[i][i] += lam

    beta = solve_linear(XtX, Xty)
    return dict(zip(feature_models, beta)), n_rows, n_pos


def main():
    ease = load_user(EASE_PATH)
    hstu = load_user(HSTU_PATH)
    sasrec = load_i2i(SASREC_PATH)
    lightfm = load_i2i(LIGHTFM_PATH)

    users = set(ease) | set(hstu)
    print(f"Training blender on {len(users)} users")

    # Precompute per-user top-K lists once.
    tops_by_user = {}
    for u in users:
        tops = user_top_lists(u, ease, hstu, sasrec, lightfm)
        if tops:
            tops_by_user[u] = tops
    print(f"Users with ≥1 model: {len(tops_by_user)}")

    all_coefs = defaultdict(list)
    per_oracle = {}

    for oracle in MODELS:
        coefs, n_rows, n_pos = leave_one_out_regression(
            oracle, tops_by_user.keys(), tops_by_user
        )
        per_oracle[oracle] = {
            "coefs": {m: round(v, 6) for m, v in coefs.items()},
            "rows": n_rows,
            "positives": n_pos,
            "positive_rate": round(n_pos / max(n_rows, 1), 4),
        }
        print(f"[oracle={oracle}] rows={n_rows} "
              f"positives={n_pos} coefs={coefs}")
        for m, v in coefs.items():
            all_coefs[m].append(v)

    # Average each model's coefficient over the three oracles where it
    # served as a feature. This is its learned "usefulness" as a predictor
    # of the others.
    avg = {m: sum(all_coefs[m]) / len(all_coefs[m]) for m in MODELS}
    print(f"Averaged coefficients: {avg}")

    clipped = {m: max(0.0, v) for m, v in avg.items()}
    max_v = max(clipped.values()) or 1.0
    weights = {m: round(v / max_v, 4) for m, v in clipped.items()}
    print(f"Normalised learned weights: {weights}")

    with open(WEIGHTS_PATH, "w") as f:
        json.dump(
            {
                "weights": weights,
                "averaged_raw": {m: round(v, 6) for m, v in avg.items()},
                "per_oracle": per_oracle,
                "feature": "position_weight = 1 / log2(2 + pos), 0 if absent",
                "target": "binary: candidate is in held-out oracle top-K",
                "procedure": "leave-one-out OLS; final weight = mean β_m over oracles where m was a feature",
                "top_k": TOP_K,
                "anchor_k": ANCHOR_K,
            },
            f,
            indent=2,
        )
    print(f"Saved -> {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
