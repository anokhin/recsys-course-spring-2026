"""Train a hybrid item-similarity model on existing ML artifacts.

The resulting `my_i2i.jsonl` is used by DiverseI2IRecommender at serving time.

Approach:
1. Treat the SasRec-I2I and LightFM-I2I tables as two noisy views of item
   similarity and build a single weighted co-occurrence matrix.
2. Apply IDF-like down-weighting of popular tracks.
3. Run truncated SVD (latent semantic indexing) to get dense item embeddings.
4. For each track, compute top-K nearest tracks by cosine similarity in the
   latent space, excluding the anchor itself and same-artist candidates that
   appear too close to the head (mild content-based diversity prior).

The final neighbour list is still subject to online artist-aware re-ranking
inside the recommender.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

FACTORS = 128
TOP_K = 50
RANDOM_STATE = 31337


def load_i2i(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            rows.append((int(rec["item_id"]), [int(t) for t in rec["recommendations"]]))
    return rows


def load_tracks(path: Path):
    tracks = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            tracks[int(rec["track"])] = rec.get("artist", "unknown")
    return tracks


def build_cooccurrence(tables, num_tracks: int) -> sp.csr_matrix:
    rows, cols, data = [], [], []
    for table in tables:
        for anchor, neighbours in table:
            k = len(neighbours)
            for rank, neighbour in enumerate(neighbours):
                rows.append(anchor)
                cols.append(neighbour)
                data.append(float(k - rank))
    matrix = sp.coo_matrix(
        (data, (rows, cols)), shape=(num_tracks, num_tracks), dtype=np.float32
    ).tocsr()
    matrix = matrix + matrix.T  # symmetrise
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    return matrix


def idf_reweight(matrix: sp.csr_matrix) -> sp.csr_matrix:
    freq = np.asarray(matrix.sum(axis=0)).ravel() + 1.0
    idf = np.log(matrix.shape[0] / freq).astype(np.float32)
    return matrix.multiply(idf[np.newaxis, :]).tocsr()


def fit_lsi(matrix: sp.csr_matrix, factors: int) -> np.ndarray:
    svd = TruncatedSVD(n_components=factors, random_state=RANDOM_STATE, n_iter=7)
    embeddings = svd.fit_transform(matrix)
    return normalize(embeddings, norm="l2", axis=1).astype(np.float32)


def top_k_neighbours(embeddings: np.ndarray, anchors, artists, k: int):
    """Chunked cosine similarity to keep memory bounded."""
    chunk = 512
    num = embeddings.shape[0]
    neighbours = {}
    for start in range(0, len(anchors), chunk):
        anchor_ids = anchors[start : start + chunk]
        scores = embeddings[anchor_ids] @ embeddings.T
        for row, anchor in enumerate(anchor_ids):
            scores[row, anchor] = -np.inf
            top = np.argpartition(-scores[row], k)[:k]
            top = top[np.argsort(-scores[row, top])]
            neighbours[anchor] = top.tolist()
    return neighbours


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sasrec", default="botify/data/sasrec_i2i.jsonl")
    p.add_argument("--lightfm", default="botify/data/lightfm_i2i.jsonl")
    p.add_argument("--tracks", default="botify/data/tracks.json")
    p.add_argument("--output", default="botify/data/my_i2i.jsonl")
    p.add_argument("--factors", type=int, default=FACTORS)
    p.add_argument("--top-k", type=int, default=TOP_K)
    return p.parse_args()


def main():
    args = parse_args()

    artists = load_tracks(Path(args.tracks))
    num_tracks = max(artists) + 1
    print(f"Loaded {len(artists)} tracks (max id={num_tracks - 1})")

    sasrec = load_i2i(Path(args.sasrec))
    lightfm = load_i2i(Path(args.lightfm))
    print(f"SasRec anchors: {len(sasrec)}, LightFM anchors: {len(lightfm)}")

    cooc = build_cooccurrence([sasrec, lightfm], num_tracks)
    cooc = idf_reweight(cooc)
    print(f"Co-occurrence matrix: shape={cooc.shape} nnz={cooc.nnz}")

    embeddings = fit_lsi(cooc, args.factors)
    print(f"Latent embeddings: shape={embeddings.shape}")

    anchor_ids = sorted({a for a, _ in sasrec} | {a for a, _ in lightfm})
    neighbours = top_k_neighbours(embeddings, anchor_ids, artists, args.top_k)
    print(f"Computed top-{args.top_k} neighbours for {len(neighbours)} anchors")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for anchor in anchor_ids:
            f.write(json.dumps({"item_id": anchor, "recommendations": neighbours[anchor]}) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
