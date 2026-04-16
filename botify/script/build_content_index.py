#!/usr/bin/env python3
"""
Build content-based item-to-item index from track metadata.

Reads tracks.json, extracts content features (genres, mood, year, artist_fans),
computes cosine similarity between all tracks, and saves top-K neighbors
per track to content_i2i.jsonl.
"""

import json
import sys
import os
import numpy as np
from collections import defaultdict


def load_tracks(path):
    tracks = []
    with open(path) as f:
        for line in f:
            tracks.append(json.loads(line))
    return tracks


def build_feature_matrix(tracks):
    """Build a feature matrix from track metadata."""

    # Collect all possible genre and mood values
    all_genres = sorted({g for t in tracks for g in t.get("genres", [])})
    all_moods = sorted({t["mood"] for t in tracks if isinstance(t.get("mood"), str)
                        and t["mood"] != "No mood could be determined"})
    all_artist_genres = sorted({t["artist_genre"] for t in tracks
                                if isinstance(t.get("artist_genre"), str)})

    genre_idx = {g: i for i, g in enumerate(all_genres)}
    mood_idx = {m: i for i, m in enumerate(all_moods)}
    artist_genre_idx = {ag: i for i, ag in enumerate(all_artist_genres)}

    n_tracks = len(tracks)
    n_genres = len(all_genres)
    n_moods = len(all_moods)
    n_artist_genres = len(all_artist_genres)

    # Feature vector: genres (multi-hot) + mood (one-hot) + artist_genre (one-hot) + year + fans
    n_features = n_genres + n_moods + n_artist_genres + 2
    print(f"Feature dimensions: {n_genres} genres + {n_moods} moods + "
          f"{n_artist_genres} artist_genres + 2 numeric = {n_features}")

    matrix = np.zeros((n_tracks, n_features), dtype=np.float32)

    for i, track in enumerate(tracks):
        offset = 0

        # Genres (multi-hot, weighted)
        for g in track.get("genres", []):
            if g in genre_idx:
                matrix[i, offset + genre_idx[g]] = 1.0
        offset += n_genres

        # Mood (one-hot)
        mood = track.get("mood", "")
        if mood in mood_idx:
            matrix[i, offset + mood_idx[mood]] = 1.0
        offset += n_moods

        # Artist genre (one-hot)
        ag = track.get("artist_genre", "")
        if ag in artist_genre_idx:
            matrix[i, offset + artist_genre_idx[ag]] = 1.0
        offset += n_artist_genres

        # Year (normalized)
        year = track.get("year", 2000)
        if isinstance(year, (int, float)) and year > 0:
            matrix[i, offset] = (year - 1950) / 75.0
        else:
            matrix[i, offset] = 0.5
        offset += 1

        # Artist fans (log-normalized)
        fans = track.get("artist_fans", 50.0)
        if isinstance(fans, (int, float)):
            matrix[i, offset] = np.log1p(fans) / np.log1p(100.0)
        else:
            matrix[i, offset] = 0.5

    return matrix


def cosine_similarity_topk(matrix, top_k=20, batch_size=500):
    """Compute top-K cosine similar items for each item using batched approach."""
    n = matrix.shape[0]

    # Normalize rows to unit length
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = matrix / norms

    results = {}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = normed[start:end]  # (batch_size, features)

        # Cosine similarity: batch × all
        sims = batch @ normed.T  # (batch_size, n)

        for i in range(end - start):
            global_i = start + i
            sim_row = sims[i].copy()
            sim_row[global_i] = -1.0  # exclude self

            # Get top-K indices
            top_indices = np.argpartition(sim_row, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(-sim_row[top_indices])]

            results[global_i] = top_indices.tolist()

        if (end) % 5000 == 0 or end == n:
            print(f"  Processed {end}/{n} tracks")

    return results


def main():
    tracks_path = os.path.join(os.path.dirname(__file__), "..", "data", "tracks.json")
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "content_i2i.jsonl")

    if len(sys.argv) > 1:
        tracks_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    print(f"Loading tracks from {tracks_path}")
    tracks = load_tracks(tracks_path)
    print(f"Loaded {len(tracks)} tracks")

    print("Building feature matrix...")
    matrix = build_feature_matrix(tracks)
    print(f"Feature matrix shape: {matrix.shape}")

    print("Computing top-20 cosine neighbors per track...")
    top_k = 20
    neighbors = cosine_similarity_topk(matrix, top_k=top_k)

    print(f"Writing content i2i to {output_path}")
    with open(output_path, "w") as f:
        for track_id in sorted(neighbors.keys()):
            obj = {
                "item_id": track_id,
                "recommendations": neighbors[track_id],
            }
            f.write(json.dumps(obj) + "\n")

    print(f"Done! {len(neighbors)} items written.")


if __name__ == "__main__":
    main()
