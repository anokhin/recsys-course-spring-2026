#!/usr/bin/env python3
"""ML-style reranker that blends per-user candidates from four upstream models.

Sources
-------
- EASE  (per-user ranked tracks: recommendations_ease.json)
- HSTU  (per-user ranked tracks: hstu_recommendations.json)
- SASRec (item-to-item:        sasrec_i2i.jsonl)
- LightFM (item-to-item:       lightfm_i2i.jsonl)

Scoring
-------
For every user we build a top-K list from each model, then combine them as:

    score(track) = Σ_model  W_model * pos_weight(rank_in_model)

SASRec is the strongest signal (learned sequential model), so it gets the
highest model weight; EASE/HSTU follow; LightFM has the lowest weight.

Positional decay uses the DCG kernel (1 / log2(2 + pos)) so the top positions
dominate but tail items still contribute.

For the i2i sources (SASRec / LightFM) we derive a per-user top-K list by
using the user's top-ANCHOR_K items from EASE (or HSTU as fallback) as
anchors and aggregating i2i neighbours weighted by both the anchor position
and the neighbour position.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "botify" / "data"

EASE_PATH = DATA_DIR / "recommendations_ease.json"
HSTU_PATH = DATA_DIR / "hstu_recommendations.json"
SASREC_PATH = DATA_DIR / "sasrec_i2i.jsonl"
LIGHTFM_PATH = DATA_DIR / "lightfm_i2i.jsonl"
OUT_PATH = DATA_DIR / "recommendations_reranker.json"
WEIGHTS_PATH = Path(__file__).resolve().parent / "blender_weights.json"

# Fallback weights used when blender_weights.json is absent. Kept aligned
# with the domain prior (SASRec strongest) so the reranker is still sane
# if training is skipped.
FALLBACK_WEIGHTS = {
    "sasrec": 1.0,
    "ease": 0.6,
    "hstu": 0.5,
    "lightfm": 0.35,
}
TOP_K = 10
ANCHOR_K = 5


def load_weights():
    if WEIGHTS_PATH.exists():
        with open(WEIGHTS_PATH) as f:
            return json.load(f)["weights"]
    return FALLBACK_WEIGHTS


def position_weight(pos: int) -> float:
    return 1.0 / math.log2(2 + pos)


def load_user_rankings(path: Path) -> dict:
    users = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            users[d["user"]] = d["tracks"]
    return users


def load_i2i(path: Path) -> dict:
    i2i = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            i2i[d["item_id"]] = d["recommendations"]
    return i2i


def top_k_from_i2i(anchors, i2i, k=TOP_K):
    scores = defaultdict(float)
    anchor_set = set(anchors)
    for a_idx, anchor in enumerate(anchors):
        neighbours = i2i.get(anchor)
        if not neighbours:
            continue
        a_w = position_weight(a_idx)
        for n_idx, cand in enumerate(neighbours):
            if cand in anchor_set:
                continue
            scores[cand] += a_w * position_weight(n_idx)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [track for track, _ in ranked[:k]]


def rerank_user(user_id, ease_map, hstu_map, sasrec_i2i, lightfm_i2i,
                weights):
    per_model = {}

    ease_top = ease_map.get(user_id, [])[:TOP_K]
    if ease_top:
        per_model["ease"] = ease_top

    hstu_top = hstu_map.get(user_id, [])[:TOP_K]
    if hstu_top:
        per_model["hstu"] = hstu_top

    anchors = (ease_map.get(user_id) or hstu_map.get(user_id) or [])[:ANCHOR_K]
    if anchors:
        sas = top_k_from_i2i(anchors, sasrec_i2i)
        if sas:
            per_model["sasrec"] = sas
        lfm = top_k_from_i2i(anchors, lightfm_i2i)
        if lfm:
            per_model["lightfm"] = lfm

    if not per_model:
        return []

    combined = defaultdict(float)
    for model_name, top_list in per_model.items():
        m_w = weights[model_name]
        for pos, track in enumerate(top_list):
            combined[track] += m_w * position_weight(pos)

    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    return [track for track, _ in ranked[:TOP_K]]


def main():
    ease = load_user_rankings(EASE_PATH)
    hstu = load_user_rankings(HSTU_PATH)
    sasrec = load_i2i(SASREC_PATH)
    lightfm = load_i2i(LIGHTFM_PATH)
    weights = load_weights()

    all_users = set(ease) | set(hstu)
    print(
        f"Reranking for {len(all_users)} users "
        f"(ease={len(ease)}, hstu={len(hstu)}, "
        f"sasrec_items={len(sasrec)}, lightfm_items={len(lightfm)})"
    )
    source = "learned" if WEIGHTS_PATH.exists() else "fallback"
    print(f"Model weights ({source}): {weights}")

    written = 0
    with open(OUT_PATH, "w") as f:
        for user in sorted(all_users):
            top = rerank_user(user, ease, hstu, sasrec, lightfm, weights)
            if not top:
                continue
            f.write(json.dumps({"user": user, "tracks": top}) + "\n")
            written += 1

    print(f"Wrote {written} user rankings to {OUT_PATH}")


if __name__ == "__main__":
    main()
