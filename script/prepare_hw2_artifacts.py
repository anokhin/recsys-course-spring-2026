import argparse
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--sasrec", required=True)
    parser.add_argument("--lightfm", required=True)
    parser.add_argument("--hstu", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_tracks(path: Path):
    tracks = [json.loads(line) for line in path.open()]
    tracks = sorted(tracks, key=lambda x: int(x["track"]))
    assert [track["track"] for track in tracks] == list(range(len(tracks)))

    texts = []
    artist_ids = []
    artist_fans = []
    for track in tracks:
        year_match = re.search(r"\d{4}", str(track.get("year") or ""))
        year_token = year_match.group(0) if year_match else ""
        text = " ".join(
            filter(
                None,
                [
                    track.get("title") or "",
                    track.get("alternative_title") or "",
                    track.get("artist") or "",
                    track.get("alternative_artist") or "",
                    track.get("artist_country") or "",
                    track.get("artist_genre") or "",
                    track.get("mood") or "",
                    " ".join(track.get("genres") or []),
                    " ".join(track.get("artist_genres") or []),
                    year_token,
                    track.get("summary") or "",
                ],
            )
        )
        texts.append(text)
        artist_ids.append(int(track.get("artist_id", -1)))
        artist_fans.append(float(track.get("artist_fans") or 0.0))

    return texts, np.asarray(artist_ids, dtype=np.int32), np.asarray(artist_fans, dtype=np.float32)


def load_item_neighbors(path: Path, n_items: int):
    neighbors = [[] for _ in range(n_items)]
    edge_rows = []
    edge_cols = []
    edge_vals = []
    incoming = Counter()

    for line in path.open():
        row = json.loads(line)
        item = int(row["item_id"])
        recs = [int(x) for x in row["recommendations"]]
        neighbors[item] = recs
        for rank, candidate in enumerate(recs):
            w = 1.0 / (rank + 1)
            edge_rows.append(item)
            edge_cols.append(candidate)
            edge_vals.append(w)
            incoming[candidate] += 1

    adjacency = sparse.csr_matrix(
        (edge_vals, (edge_rows, edge_cols)), shape=(n_items, n_items), dtype=np.float32
    )
    return neighbors, adjacency, incoming


def load_hstu(path: Path):
    data = {}
    for line in path.open():
        row = json.loads(line)
        data[int(row["user"])] = [int(x) for x in row["tracks"]]
    return data


def build_text_embedding(texts):
    vectorizer = TfidfVectorizer(
        max_features=12000, ngram_range=(1, 2), min_df=2, max_df=0.9
    )
    tfidf = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=32, random_state=42)
    emb = svd.fit_transform(tfidf).astype(np.float32)
    return emb


def build_graph_embedding(
    sasrec_adj: sparse.csr_matrix, lightfm_adj: sparse.csr_matrix
) -> np.ndarray:
    fused = sasrec_adj + 0.7 * lightfm_adj
    fused = fused + fused.T
    svd = TruncatedSVD(n_components=32, random_state=42)
    emb = svd.fit_transform(fused).astype(np.float32)
    return emb


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def build_hybrid_neighbors(graph_emb: np.ndarray, text_emb: np.ndarray):
    graph_norm = normalize(graph_emb)
    text_norm = normalize(text_emb)
    hybrid = np.hstack([math.sqrt(0.78) * graph_norm, math.sqrt(0.22) * text_norm]).astype(
        np.float32
    )
    nn = NearestNeighbors(n_neighbors=31, metric="cosine", algorithm="brute")
    nn.fit(hybrid)
    distances, indices = nn.kneighbors(hybrid, return_distance=True)
    neighbors = [list(map(int, row[1:])) for row in indices]
    return neighbors


def build_popularity(
    n_items: int,
    sasrec_incoming: Counter,
    lightfm_incoming: Counter,
    artist_fans: np.ndarray,
) -> np.ndarray:
    score = np.zeros(n_items, dtype=np.float32)
    for item, cnt in sasrec_incoming.items():
        score[item] += 1.0 * cnt
    for item, cnt in lightfm_incoming.items():
        score[item] += 0.7 * cnt

    if artist_fans.size:
        score += 0.05 * np.log1p(artist_fans)

    score -= score.min()
    score /= score.max() + 1e-8
    return score.astype(np.float32)


def main():
    args = parse_args()

    texts, artist_ids, artist_fans = load_tracks(Path(args.tracks))
    n_items = len(texts)

    sasrec_neighbors, sasrec_adj, sasrec_incoming = load_item_neighbors(
        Path(args.sasrec), n_items
    )
    lightfm_neighbors, lightfm_adj, lightfm_incoming = load_item_neighbors(
        Path(args.lightfm), n_items
    )
    hstu_candidates = load_hstu(Path(args.hstu))

    text_emb = build_text_embedding(texts)
    graph_emb = build_graph_embedding(sasrec_adj, lightfm_adj)
    hybrid_neighbors = build_hybrid_neighbors(graph_emb, text_emb)
    popularity = build_popularity(
        n_items, sasrec_incoming, lightfm_incoming, artist_fans
    )
    popular_tracks = np.argsort(-popularity)[:300].astype(np.int32).tolist()

    artifact = {
        "graph_emb": graph_emb.astype(np.float32),
        "text_emb": text_emb.astype(np.float32),
        "artist_ids": artist_ids.astype(np.int32),
        "hybrid_neighbors": hybrid_neighbors,
        "sasrec_neighbors": sasrec_neighbors,
        "lightfm_neighbors": lightfm_neighbors,
        "hstu_candidates": hstu_candidates,
        "popularity": popularity.astype(np.float32),
        "popular_tracks": popular_tracks,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"saved {output}")


if __name__ == "__main__":
    main()
