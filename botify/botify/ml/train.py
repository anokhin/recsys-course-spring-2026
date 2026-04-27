"""Offline trainer for the homework-2 reranker artifacts.

The trainer is executed once during the Docker image build. It consumes
the static catalog files shipped with the service and produces:

* ``track_embeddings.npy``   - dense item embeddings (Truncated SVD on the
                                union of the SasRec-I2I and LightFM-I2I
                                co-occurrence graphs).
* ``track_index.json``       - mapping from track id to embedding row.
* ``artist_index.json``      - mapping from track id to artist id.

The script is deterministic (fixed ``random_state``) which is critical
for the reproducibility check performed by the course grader.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

LOGGER = logging.getLogger("botify.ml.train")

EMBEDDING_DIM = 64
SVD_RANDOM_STATE = 42
SASREC_WEIGHT = 1.0
LIGHTFM_WEIGHT = 0.6


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_track_catalog(catalog_path: Path) -> Tuple[Dict[int, int], int]:
    """Return ``(artist_by_track, max_track_id)`` parsed from the catalog."""
    artist_by_track: Dict[int, int] = {}
    max_id = -1
    for entry in _read_jsonl(catalog_path):
        track_id = int(entry["track"])
        artist_id = int(entry["artist_id"])
        artist_by_track[track_id] = artist_id
        max_id = max(max_id, track_id)
    if not artist_by_track:
        raise RuntimeError(f"Catalog {catalog_path} is empty")
    return artist_by_track, max_id


def _build_edges(
    path: Path, weight: float
) -> Tuple[List[int], List[int], List[float]]:
    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []
    for entry in _read_jsonl(path):
        src = int(entry["item_id"])
        for rank, dst in enumerate(entry["recommendations"]):
            dst = int(dst)
            if dst == src:
                continue
            rows.append(src)
            cols.append(dst)
            values.append(weight / np.log2(2.0 + rank))
    return rows, cols, values


def build_cooccurrence(
    n_tracks: int,
    sources: List[Tuple[Path, float]],
) -> sp.csr_matrix:
    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []
    for path, weight in sources:
        if not path.exists():
            LOGGER.warning("I2I source missing: %s", path)
            continue
        r, c, v = _build_edges(path, weight)
        rows.extend(r)
        cols.extend(c)
        values.extend(v)

    if not rows:
        raise RuntimeError("No i2i edges found - cannot train embeddings")

    matrix = sp.coo_matrix(
        (values, (rows, cols)), shape=(n_tracks, n_tracks), dtype=np.float32
    ).tocsr()
    matrix = matrix.maximum(matrix.T)
    return matrix


def train_embeddings(matrix: sp.csr_matrix, dim: int) -> np.ndarray:
    effective_dim = min(dim, matrix.shape[0] - 1)
    svd = TruncatedSVD(
        n_components=effective_dim,
        random_state=SVD_RANDOM_STATE,
        algorithm="randomized",
        n_iter=7,
    )
    embeddings = svd.fit_transform(matrix).astype(np.float32, copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return (embeddings / norms).astype(np.float32, copy=False)


def save_artifacts(
    output_dir: Path,
    embeddings: np.ndarray,
    artist_by_track: Dict[int, int],
    n_tracks: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "track_embeddings.npy", embeddings)

    track_index = {str(track_id): track_id for track_id in range(n_tracks)}
    with (output_dir / "track_index.json").open("w", encoding="utf-8") as fp:
        json.dump(track_index, fp)

    serialisable_artists = {
        str(track_id): int(artist) for track_id, artist in artist_by_track.items()
    }
    with (output_dir / "artist_index.json").open("w", encoding="utf-8") as fp:
        json.dump(serialisable_artists, fp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dim", type=int, default=EMBEDDING_DIM)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    artist_by_track, max_id = load_track_catalog(args.data_dir / "tracks.json")
    n_tracks = max_id + 1
    LOGGER.info("Loaded %d tracks (max id %d)", len(artist_by_track), max_id)

    sources = [
        (args.data_dir / "sasrec_i2i.jsonl", SASREC_WEIGHT),
        (args.data_dir / "lightfm_i2i.jsonl", LIGHTFM_WEIGHT),
    ]
    matrix = build_cooccurrence(n_tracks, sources)
    LOGGER.info(
        "Co-occurrence matrix: shape=%s nnz=%d",
        matrix.shape,
        matrix.nnz,
    )

    embeddings = train_embeddings(matrix, args.dim)
    LOGGER.info("Trained embeddings: shape=%s", embeddings.shape)

    save_artifacts(args.output_dir, embeddings, artist_by_track, n_tracks)
    LOGGER.info("Saved artifacts to %s", args.output_dir)


if __name__ == "__main__":
    main()
