#!/usr/bin/env python3
"""Bootstrap I2I без gensim: DeepWalk-cooccurrence + SVD + content features.

Учимся на графе LightFM∪SasRec и метаданных tracks.json (не sim/).
На выходе — track2vec_i2i.jsonl для DiverseI2IRecommender.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize


def load_i2i_graph(path: Path) -> dict[int, list[int]]:
    graph = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            item = int(row["item_id"])
            graph[item] = [int(x) for x in row["recommendations"]]
    return graph


def load_tracks(path: Path):
    tracks = []
    with path.open() as f:
        for line in f:
            tracks.append(json.loads(line))
    return tracks


def build_cooccurrence(graph: dict[int, list[int]], n_walks: int, walk_len: int, seed: int):
    """Считаем co-occurrence по случайным блужданиям (DeepWalk-style)."""
    rng = random.Random(seed)
    # В индекс попадают и ключи, и соседи — иначе KeyError на «висячих» item_id
    all_nodes = set(graph.keys())
    for recs in graph.values():
        all_nodes.update(recs)
    nodes = sorted(all_nodes)
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    counts = lil_matrix((n, n), dtype=np.float32)

    # Для узлов без исходящих рёбер — пустой список
    safe_graph = {n: [x for x in (graph.get(n) or []) if x in index] for n in nodes}

    for start in nodes:
        for _ in range(n_walks):
            walk = [start]
            cur = start
            for _ in range(walk_len - 1):
                neigh = safe_graph.get(cur) or []
                if not neigh:
                    break
                cur = rng.choice(neigh)
                walk.append(cur)
            for i, a in enumerate(walk):
                ia = index[a]
                for j in range(i + 1, min(i + 3, len(walk))):
                    b = walk[j]
                    ib = index[b]
                    if ia == ib:
                        continue
                    counts[ia, ib] += 1.0
                    counts[ib, ia] += 1.0
    return nodes, counts.tocsr()


def content_embeddings(tracks, dim: int, seed: int):
    feats = []
    ids = []
    for t in tracks:
        ids.append(int(t["track"]))
        row = {
            f"artist={t.get('artist')}": 1.0,
            f"mood={t.get('mood')}": 1.0,
            f"year={t.get('year')}": 1.0,
        }
        for g in t.get("genres") or []:
            row[f"genre={g}"] = 1.0
        for g in t.get("artist_genres") or []:
            row[f"agenre={g}"] = 0.5
        feats.append(row)

    X = DictVectorizer().fit_transform(feats)
    n_comp = min(dim, max(2, X.shape[1] - 1), X.shape[0] - 1)
    emb = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(X)
    emb = normalize(emb)
    return {tid: emb[i] for i, tid in enumerate(ids)}, emb.shape[1]


def diversify(neighbours, track2artist, top_k, max_per_artist):
    picked = []
    cnt = defaultdict(int)
    for tid in neighbours:
        a = track2artist.get(tid, "")
        if cnt[a] >= max_per_artist:
            continue
        picked.append(int(tid))
        cnt[a] += 1
        if len(picked) >= top_k:
            break
    return picked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lfm-i2i", default="botify/data/lightfm_i2i.jsonl")
    p.add_argument("--sasrec-i2i", default="botify/data/sasrec_i2i.jsonl")
    p.add_argument("--tracks", default="botify/data/tracks.json")
    p.add_argument("--output", default="botify/data/track2vec_i2i.jsonl")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--walks", type=int, default=8)
    p.add_argument("--walk-len", type=int, default=16)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-per-artist", type=int, default=2)
    p.add_argument("--content-weight", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    graph = load_i2i_graph(Path(args.lfm_i2i))
    sas = load_i2i_graph(Path(args.sasrec_i2i))
    for item, recs in sas.items():
        graph[item] = list(dict.fromkeys((graph.get(item) or []) + recs))

    tracks = load_tracks(Path(args.tracks))
    track2artist = {int(t["track"]): str(t["artist"]) for t in tracks}
    print(f"Union graph nodes: {len(graph)}, tracks: {len(tracks)}")

    nodes, cooc = build_cooccurrence(
        graph, n_walks=args.walks, walk_len=args.walk_len, seed=args.seed
    )
    print(f"Co-occurrence nnz: {cooc.nnz}")

    # PMI-like: log1p counts, затем SVD
    cooc.data = np.log1p(cooc.data)
    n_comp = min(args.dim, max(2, len(nodes) - 1))
    cf = TruncatedSVD(n_components=n_comp, random_state=args.seed).fit_transform(cooc)
    cf = normalize(cf.astype(np.float32))

    content_map, cdim = content_embeddings(tracks, dim=args.dim, seed=args.seed)
    content_mat = np.zeros_like(cf)
    for i, tid in enumerate(nodes):
        vec = content_map.get(tid)
        if vec is None:
            continue
        v = np.zeros(cf.shape[1], dtype=np.float32)
        n = min(len(vec), cf.shape[1])
        v[:n] = vec[:n]
        content_mat[i] = v
    content_mat = normalize(content_mat)

    w = args.content_weight
    blended = normalize((1.0 - w) * cf + w * content_mat)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pool = max(200, args.top_k * 6)

    # Попарный поиск соседей батчами по query
    with out.open("w") as f:
        batch = 256
        for start in range(0, len(nodes), batch):
            end = min(start + batch, len(nodes))
            sims = blended[start:end] @ blended.T  # (B, N)
            for local_i, i in enumerate(range(start, end)):
                row = sims[local_i]
                row[i] = -1.0
                cand_idx = np.argpartition(row, -pool)[-pool:]
                cand_idx = cand_idx[np.argsort(row[cand_idx])[::-1]]
                neighbours = [nodes[j] for j in cand_idx]
                neighbours = diversify(
                    neighbours, track2artist, args.top_k, args.max_per_artist
                )
                f.write(
                    json.dumps(
                        {"item_id": int(nodes[i]), "recommendations": neighbours}
                    )
                    + "\n"
                )
            if end % 2048 == 0 or end == len(nodes):
                print(f"exported {end}/{len(nodes)}")

    print(f"Saved → {out} ({len(nodes)} items)")


if __name__ == "__main__":
    main()
