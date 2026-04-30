import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


CATALOG_PATH = Path("botify/data/tracks.json")
OUT_PATH = Path("botify/data/learned_i2i.jsonl")

TOPK = 100
N_COMPONENTS = 128


def load_catalog():
    rows = []

    with open(CATALOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            r = json.loads(line)

            track_id = int(r["track"])

            parts = []
            for key in ["artist", "genres", "genre", "mood", "summary", "name", "title"]:
                val = r.get(key)

                if val is None:
                    continue

                if isinstance(val, list):
                    parts.extend(map(str, val))
                else:
                    parts.append(str(val))

            text = " ".join(parts)

            rows.append({
                "track_id": track_id,
                "text": text,
            })

    return pd.DataFrame(rows)


def main():
    df = load_catalog()

    print("tracks:", len(df))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
    )

    X = vectorizer.fit_transform(df["text"].fillna(""))

    n_components = min(N_COMPONENTS, X.shape[1] - 1)

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42,
    )

    Z = svd.fit_transform(X)
    Z = normalize(Z)

    sim = Z @ Z.T
    np.fill_diagonal(sim, -1e9)

    track_ids = df["track_id"].astype(int).tolist()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w") as f:
        for i, track_id in enumerate(track_ids):
            order = np.argsort(-sim[i])[:TOPK]
            recs = [int(track_ids[j]) for j in order]

            f.write(json.dumps({
                "item_id": int(track_id),
                "recommendations": recs,
            }) + "\n")

    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
