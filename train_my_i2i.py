import json
import glob
import math
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


LOG_GLOB = "train_logs/**/data.json*"
TRACKS_PATH = "botify/data/tracks.json"
OUTPUT = "botify/data/my_i2i.jsonl"

TOPK = 50
NEG_PER_POS = 20
RANDOM_SEED = 42
GOOD_TIME = 0.65
MAX_TRAIN = 1_500_000


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue


def load_tracks():
    meta = {}
    for row in read_jsonl(TRACKS_PATH):
        track = int(row["track"])
        meta[track] = {
            "artist_id": row.get("artist_id"),
            "genre": row.get("artist_genre"),
            "mood": row.get("mood"),
            "year": row.get("year"),
            "fans": float(row.get("artist_fans", 0.0) or 0.0),
        }
    return meta


def read_logs():
    for path in glob.glob(LOG_GLOB, recursive=True):
        for row in read_jsonl(path):
            if row.get("message") not in ("next", "last"):
                continue
            if row.get("user") is None or row.get("track") is None:
                continue
            yield {
                "user": int(row["user"]),
                "track": int(row["track"]),
                "recommendation": row.get("recommendation"),
                "time": float(row.get("time", 0.0)),
                "timestamp": int(row.get("timestamp", 0)),
            }


def make_features(prev, cand, meta, transitions, popularity):
    a = meta.get(prev, {})
    b = meta.get(cand, {})

    same_artist = 1.0 if a.get("artist_id") == b.get("artist_id") and a.get("artist_id") is not None else 0.0
    same_genre = 1.0 if a.get("genre") == b.get("genre") and a.get("genre") is not None else 0.0
    same_mood = 1.0 if a.get("mood") == b.get("mood") and a.get("mood") is not None else 0.0

    try:
        year_diff = abs(float(a.get("year") or 0) - float(b.get("year") or 0))
    except Exception:
        year_diff = 999.0

    year_close = 1.0 / (1.0 + year_diff / 10.0)
    cooc = float(transitions.get(prev, Counter()).get(cand, 0.0))
    pop = math.log1p(float(popularity.get(cand, 0.0)))
    fans = float(b.get("fans", 0.0)) / 100.0

    return [
        cooc,
        pop,
        same_artist,
        same_genre,
        same_mood,
        year_close,
        fans,
    ]


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    meta = load_tracks()
    all_tracks = list(meta.keys())

    by_user = defaultdict(list)

    for row in read_logs():
        by_user[row["user"]].append(row)

    transitions = defaultdict(Counter)
    popularity = Counter()
    positives = []

    for user, events in by_user.items():
        events.sort(key=lambda x: x["timestamp"])

        for i in range(len(events) - 1):
            cur = events[i]
            nxt = events[i + 1]

            prev = int(cur["track"])
            cand = int(nxt["track"])
            next_time = float(nxt.get("time", 0.0))

            if prev == cand:
                continue
            if next_time < GOOD_TIME:
                continue

            weight = 1.0 + 3.0 * min(1.0, next_time)

            if cur.get("recommendation") is not None:
                try:
                    if int(cur["recommendation"]) == cand:
                        weight *= 1.5
                except Exception:
                    pass

            transitions[prev][cand] += weight
            popularity[cand] += weight
            positives.append((prev, cand))

    print("positive transitions:", len(positives), flush=True)

    popular_tracks = [t for t, _ in popularity.most_common(500)]

    X = []
    y = []

    for prev, pos in positives:
        X.append(make_features(prev, pos, meta, transitions, popularity))
        y.append(1)

        for _ in range(NEG_PER_POS):
            if len(X) >= MAX_TRAIN:
                break

            if popular_tracks and rng.random() < 0.8:
                neg = int(popular_tracks[int(rng.integers(0, len(popular_tracks)))])
            else:
                neg = int(all_tracks[int(rng.integers(0, len(all_tracks)))])

            if neg == prev or neg == pos:
                continue

            X.append(make_features(prev, neg, meta, transitions, popularity))
            y.append(0)

        if len(X) >= MAX_TRAIN:
            break

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    print("dataset:", X.shape, "positive:", int(y.sum()), "negative:", int((1 - y).sum()), flush=True)

    model = LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=300,
        class_weight="balanced",
        C=1.0,
    )
    model.fit(X, y)

    print("model trained", flush=True)

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

    written = 0

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for item_id in sorted(all_tracks):
            candidates = set()

            for cand, _ in transitions.get(item_id, Counter()).most_common(500):
                if cand != item_id:
                    candidates.add(int(cand))

            current = meta.get(item_id, {})

            for cand in popular_tracks[:700]:
                if cand == item_id:
                    continue

                other = meta.get(cand, {})

                if (
                    current.get("artist_id") == other.get("artist_id")
                    or current.get("genre") == other.get("genre")
                    or current.get("mood") == other.get("mood")
                ):
                    candidates.add(int(cand))

                if len(candidates) >= 700:
                    break

            if not candidates:
                continue

            cand_list = list(candidates)
            rows = [
                make_features(item_id, cand, meta, transitions, popularity)
                for cand in cand_list
            ]

            scores = model.predict_proba(np.asarray(rows, dtype=np.float32))[:, 1]
            ranked = sorted(zip(cand_list, scores), key=lambda x: x[1], reverse=True)

            recs = [int(cand) for cand, _ in ranked[:TOPK]]

            if recs:
                out.write(json.dumps({
                    "item_id": int(item_id),
                    "recommendations": recs,
                }) + "\n")
                written += 1

    print(f"Saved {OUTPUT}, rows={written}", flush=True)


if __name__ == "__main__":
    main()