import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


MAX_FEATURES = 6000
N_COMPONENTS = 64
N_NEIGHBORS = 128
RANDOM_STATE = 31312


def load_documents(tracks_path: Path):
    docs = []
    with tracks_path.open() as f:
        for line in f:
            row = json.loads(line)
            docs.append(
                " ".join(
                    [
                        row["title"],
                        row["artist"],
                        row.get("mood") or "",
                        " ".join(row.get("genres") or []),
                        " ".join(row.get("artist_genres") or []),
                        str(row.get("year") or ""),
                        row.get("summary") or "",
                    ]
                )
            )
    return docs


def build_vectors(docs):
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
    )
    tfidf = vectorizer.fit_transform(docs)
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    vectors = svd.fit_transform(tfidf).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    return vectors


def build_neighbors(vectors):
    k = N_NEIGHBORS
    neighbors = np.empty((vectors.shape[0], k), dtype=np.int32)

    for start in range(0, vectors.shape[0], 512):
        stop = min(start + 512, vectors.shape[0])
        similarities = vectors[start:stop] @ vectors.T
        idx = np.argpartition(-similarities, kth=k, axis=1)[:, : k + 1]
        part = np.take_along_axis(similarities, idx, axis=1)
        order = np.argsort(-part, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)

        for row_idx in range(stop - start):
            row = idx[row_idx]
            row = row[row != start + row_idx][:k]
            neighbors[start + row_idx] = row

    return neighbors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracks",
        default="data/tracks.json",
        help="Path to botify track catalog",
    )
    parser.add_argument(
        "--output",
        default="data/session_semantic_embeddings.npz",
        help="Where to write compressed embeddings",
    )
    args = parser.parse_args()

    tracks_path = Path(args.tracks)
    output_path = Path(args.output)

    docs = load_documents(tracks_path)
    vectors = build_vectors(docs)
    neighbors = build_neighbors(vectors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, vectors=vectors, neighbors=neighbors)
    print(output_path)


if __name__ == "__main__":
    main()
