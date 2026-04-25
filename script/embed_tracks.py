"""
Compute sentence-transformer embeddings of tracks based on their metadata
(summary, mood, genres, artist, country) and save as a contiguous numpy array
indexed by track id.

Usage:
    python script/embed_tracks.py \
        --tracks botify/data/tracks.json \
        --output botify/data/track_embeddings.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_text(track: dict) -> str:
    summary = (track.get("summary") or "").strip()
    mood = (track.get("mood") or "").strip()
    genres = ", ".join(track.get("genres") or [])
    artist = (track.get("artist") or "").strip()
    country = (track.get("artist_country") or "").strip()
    title = (track.get("title") or "").strip()
    return (
        f"Title: {title}. "
        f"Artist: {artist}. "
        f"Country: {country}. "
        f"Genres: {genres}. "
        f"Mood: {mood}. "
        f"Summary: {summary}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    tracks = []
    with open(args.tracks) as f:
        for line in f:
            tracks.append(json.loads(line))
    n = len(tracks)
    if n == 0:
        raise RuntimeError(f"No tracks in {args.tracks}")

    ids = [t["track"] for t in tracks]
    if min(ids) != 0 or max(ids) != n - 1 or len(set(ids)) != n:
        raise RuntimeError(
            f"Track ids not dense: min={min(ids)} max={max(ids)} "
            f"count={len(set(ids))} n={n}"
        )

    tracks.sort(key=lambda t: t["track"])
    texts = [build_text(t) for t in tracks]

    model = SentenceTransformer(args.model)
    embs = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embs)
    print(f"wrote {args.output}  shape={embs.shape}  dtype={embs.dtype}")


if __name__ == "__main__":
    main()
