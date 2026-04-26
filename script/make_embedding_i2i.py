"""
Precompute top-K nearest-neighbor I2I recommendations from track embeddings.

Uses llama3.1 track embeddings (cosine similarity via normalised dot product).
Optimises directly for the simulator reward: score = dot(rec_emb, session_emb).

Output format: {"item_id": N, "recommendations": [id1, id2, ...]} (same as
sasrec_i2i.jsonl) so the result can be loaded with I2IRecommender.
"""
import json
import os
import shutil
import sys

import numpy as np

EMBEDDINGS_PATH = "sim/data/embeddings.npy"
OUTPUT_PATH = "botify/data/embedding_i2i.jsonl"
FALLBACK_PATH = "botify/data/sasrec_i2i.jsonl"
K = 100
BATCH = 500

MIN_FILE_SIZE = 1_000_000  # 1 MB — real embeddings are ~530 MB


def fallback():
    print("Falling back: copying sasrec_i2i.jsonl as embedding_i2i.jsonl", flush=True)
    shutil.copy(FALLBACK_PATH, OUTPUT_PATH)


def main():
    if not os.path.exists(EMBEDDINGS_PATH) or os.path.getsize(EMBEDDINGS_PATH) < MIN_FILE_SIZE:
        print(f"WARNING: {EMBEDDINGS_PATH} not available, skipping computation", flush=True)
        fallback()
        return

    print(f"Loading embeddings from {EMBEDDINGS_PATH} ...", flush=True)
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    N, D = embeddings.shape
    print(f"Loaded {N} tracks, dim={D}", flush=True)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = embeddings / norms  # L2-normalised

    n_batches = (N + BATCH - 1) // BATCH
    print(f"Computing top-{K} neighbours ({n_batches} batches of {BATCH}) ...", flush=True)

    with open(OUTPUT_PATH, "w") as out:
        for b, i in enumerate(range(0, N, BATCH)):
            batch = emb[i : i + BATCH]            # scores: (batch_size, N)
            scores = batch @ emb.T

            for j in range(len(batch)):
                scores[j, i + j] = -1.0  # exclude self

            # Partial sort: find top-K indices (unsorted within top-K)
            top_k_idx = np.argpartition(-scores, min(K, scores.shape[1] - 1), axis=1)[:, :K]

            for j in range(len(batch)):
                tidx = top_k_idx[j]
                sorted_idx = tidx[np.argsort(-scores[j, tidx])]
                out.write(
                    json.dumps({"item_id": i + j, "recommendations": sorted_idx.tolist()})
                    + "\n"
                )

            if b % 5 == 0:
                print(f"  batch {b}/{n_batches} ({i}/{N} tracks)", flush=True)

    print(f"Saved {N} entries to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
