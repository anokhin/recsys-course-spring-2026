"""Lazy loading of pre-trained ML artifacts used by the online recommender.

The artifacts are produced by ``botify.ml.train`` during the Docker image
build and consumed by the Flask app at startup. Loading is intentionally
defensive: if the artifacts are missing or corrupted the recommender
falls back to a baseline rather than crashing the service.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ItemEmbeddings:
    """Dense item embeddings indexed by track id.

    The matrix is L2-normalised so dot product equals cosine similarity.
    """

    matrix: np.ndarray
    track_to_row: Dict[int, int]

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def get(self, track_id: int) -> Optional[np.ndarray]:
        row = self.track_to_row.get(int(track_id))
        if row is None:
            return None
        return self.matrix[row]


@dataclass(frozen=True)
class MLArtifacts:
    embeddings: ItemEmbeddings
    artist_by_track: Dict[int, int]


def _load_track_index(path: Path) -> Dict[int, int]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(track_id): int(row) for track_id, row in raw.items()}


def _load_artist_index(path: Path) -> Dict[int, int]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(track_id): int(artist_id) for track_id, artist_id in raw.items()}


def load_artifacts(artifacts_dir: str) -> Optional[MLArtifacts]:
    """Load pre-trained artifacts. Return ``None`` on any failure."""
    base = Path(artifacts_dir)
    embeddings_path = base / "track_embeddings.npy"
    track_index_path = base / "track_index.json"
    artist_index_path = base / "artist_index.json"

    missing = [
        p for p in (embeddings_path, track_index_path, artist_index_path)
        if not p.exists()
    ]
    if missing:
        LOGGER.warning("ML artifacts missing: %s", [str(p) for p in missing])
        return None

    try:
        matrix = np.load(embeddings_path).astype(np.float32, copy=False)
        track_to_row = _load_track_index(track_index_path)
        artist_by_track = _load_artist_index(artist_index_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("Failed to load ML artifacts: %s", exc)
        return None

    if matrix.ndim != 2 or matrix.shape[0] == 0:
        LOGGER.error("Invalid embedding matrix shape: %s", matrix.shape)
        return None

    embeddings = ItemEmbeddings(matrix=matrix, track_to_row=track_to_row)
    return MLArtifacts(embeddings=embeddings, artist_by_track=artist_by_track)
