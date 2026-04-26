import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TRACKS_PATH = Path("botify/data/tracks.json")
OUTPUT_PATH = Path("botify/data/content_i2i.jsonl")
TOP_K = 50
BATCH_SIZE = 512


def year_bucket(year):
    if not year:
        return "year_unknown"
    try:
        y = int(str(year).strip()[:4])
    except (ValueError, TypeError):
        return "year_unknown"
    decade = (y // 10) * 10
    return f"decade_{decade}"


def build_feature_text(track: dict) -> str:
    parts = []
    genres = track.get("genres") or []
    parts.extend(genres * 3)
    ag = track.get("artist_genre") or ""
    if ag:
        parts.extend([ag] * 2)

    mood = track.get("mood") or ""
    if mood:
        parts.extend([mood.replace(" ", "_")] * 3)

    country = track.get("artist_country") or ""
    if country:
        parts.append(country.replace(" ", "_"))

    parts.append(year_bucket(track.get("year")))
    return " ".join(parts)


def main():
    print("Loading tracks...")
    tracks = []
    with open(TRACKS_PATH) as f:
        for line in f:
            tracks.append(json.loads(line))
    print(f"  {len(tracks)} tracks loaded")

    track_ids = [t["track"] for t in tracks]
    artists = [t["artist"] for t in tracks]
    texts = [build_feature_text(t) for t in tracks]

    print("Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(min_df=1, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(texts)
    tfidf_dense = normalize(tfidf, norm="l2")

    print(f"  Matrix: {tfidf_dense.shape}, vocab={len(vectorizer.vocabulary_)}")

    print("Converting to dense for fast dot-product...")
    mat = tfidf_dense.toarray().astype(np.float32)

    print(f"Generating top-{TOP_K} content I2I (same-artist excluded)...")
    results = {}

    n = len(track_ids)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = mat[start:end]
        sims = batch @ mat.T

        for i, row_sims in enumerate(sims):
            idx = start + i
            anchor_artist = artists[idx]
            anchor_id = track_ids[idx]

            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            for j, a in enumerate(artists):
                if a == anchor_artist:
                    mask[j] = False

            valid_sims = row_sims.copy()
            valid_sims[~mask] = -1.0

            top_indices = np.argpartition(valid_sims, -TOP_K)[-TOP_K:]
            top_indices = top_indices[np.argsort(-valid_sims[top_indices])]

            results[anchor_id] = [track_ids[j] for j in top_indices]

        if (start // BATCH_SIZE) % 5 == 0:
            print(f"  {end}/{n} tracks done")

    print(f"Writing {len(results)} entries to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as out:
        for item_id, recs in results.items():
            out.write(
                json.dumps({"item_id": item_id, "recommendations": recs}) + "\n"
            )

    print("Done")
    sample_id = track_ids[0]
    sample_recs = results[sample_id]
    sample_track = tracks[0]
    print(
        f"\nSample: track={sample_id} ({sample_track['artist']} — {sample_track['title']})"
    )
    print(f"  genres={sample_track['genres']}, mood={sample_track['mood']}")
    print("  Top-5 recs:")
    tid2track = {t["track"]: t for t in tracks}
    for r in sample_recs[:5]:
        rt = tid2track[r]
        print(
            f"    {r}: {rt['artist']} — {rt['title']} | {rt['genres']} | {rt['mood']}"
        )


if __name__ == "__main__":
    main()
