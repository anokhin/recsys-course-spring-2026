from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _track_text(t: dict) -> str:
    parts = [
        t.get("artist") or "",
        t.get("artist") or "",
        t.get("artist") or "",
        " ".join(t.get("genres") or []),
        " ".join(t.get("artist_genres") or []),
        t.get("mood") or "",
        t.get("artist_country") or "",
        t.get("title") or "",
        t.get("summary") or "",
    ]
    return " ".join(p for p in parts if p)


def build(catalog_path: Path, out_path: Path, meta_path: Path, dim: int = 64) -> None:
    tracks: list[dict] = []
    with catalog_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                tracks.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  skip malformed: {e}")
    print(f"loaded {len(tracks)} tracks from {catalog_path}")

    texts = [_track_text(t) for t in tracks]
    track_ids = [int(t["track"]) for t in tracks]
    artists = [t.get("artist") or "" for t in tracks]

    n_tracks = max(track_ids) + 1
    print(f"max track id = {max(track_ids)}, n_tracks = {n_tracks}")

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.7,
        sublinear_tf=True,
        analyzer="word",
        lowercase=True,
    )
    X = vec.fit_transform(texts)
    print(f"tfidf: {X.shape[0]} x {X.shape[1]}, nnz={X.nnz}")

    svd = TruncatedSVD(n_components=min(dim, X.shape[1] - 1), random_state=42)
    Z = svd.fit_transform(X).astype(np.float32)
    Z = normalize(Z, norm="l2", axis=1).astype(np.float32)
    print(f"svd dims = {Z.shape[1]}, evr_sum = {svd.explained_variance_ratio_.sum():.3f}")

    embeddings = np.zeros((n_tracks, Z.shape[1]), dtype=np.float32)
    for row, tid in enumerate(track_ids):
        embeddings[tid] = Z[row]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    print(f"embeddings -> {out_path} shape={embeddings.shape}")

    artist_by_track = {}
    for tid, a in zip(track_ids, artists):
        artist_by_track[str(tid)] = a
    meta_path.write_text(
        json.dumps({"n_tracks": n_tracks, "dim": int(Z.shape[1]), "artist_by_track": artist_by_track})
    )
    print(f"meta -> {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "tracks.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "content_embeddings.npy",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "content_embeddings_meta.json",
    )
    parser.add_argument("--dim", type=int, default=64)
    args = parser.parse_args()
    build(args.catalog, args.out, args.meta, dim=args.dim)


if __name__ == "__main__":
    main()
