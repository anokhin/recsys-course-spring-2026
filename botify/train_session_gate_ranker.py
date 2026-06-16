import json
import pickle
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import redis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LOG_PATH = Path("../train_data_v3.json")
TRACKS_PATH = Path("./data/tracks.json")
SASREC_PATH = Path("./data/sasrec_i2i.jsonl")
LIGHTFM_PATH = Path("./data/lightfm_i2i.jsonl")
OUTPUT_PATH = Path("./session_gate_ranker_bundle.joblib")

GOOD_TIME = 0.75
SKIP_TIME = 0.20
ANCHOR_WINDOW = 4
TOPK_PER_SOURCE = 10
MAX_CANDIDATES = 40
RANDOM_SEED = 42


def load_i2i(path: Path):
    table = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            item_id = int(row["item_id"])
            recs = [int(x) for x in row["recommendations"]]
            table[item_id] = recs
    return table


def load_tracks(path: Path):
    meta = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta[int(row["track"])] = row
    return meta


def read_control_logs(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.replace("}{", "}\n{").splitlines()
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                try:
                    x = json.loads(part)
                except json.JSONDecodeError:
                    continue

                exp = x.get("experiments", {})
                if exp.get("HSTU") != "C":
                    continue
                if x.get("message") not in ("next", "last"):
                    continue

                rows.append(x)
    return rows


def build_sessions(rows):
    by_user = defaultdict(list)
    for x in rows:
        by_user[int(x["user"])].append(x)

    sessions = []
    for user, events in by_user.items():
        events.sort(key=lambda z: int(z["timestamp"]))

        current = []
        for e in events:
            current.append({
                "track": int(e["track"]),
                "time": float(e.get("time", 0.0)),
                "message": e["message"],
                "recommendation": e.get("recommendation"),
            })
            if e["message"] == "last":
                if len(current) >= 2:
                    sessions.append((user, current))
                current = []

        if len(current) >= 2:
            sessions.append((user, current))

    return sessions


def get_recent_stats(history):
    recent = history[-ANCHOR_WINDOW:]
    times = [x[1] for x in recent]

    avg_time = float(np.mean(times)) if times else 0.0
    last_time = float(times[-1]) if times else 0.0
    good_frac = float(np.mean([t >= 0.7 for t in times])) if times else 0.0
    skip_frac = float(np.mean([t < 0.2 for t in times])) if times else 0.0

    return avg_time, last_time, good_frac, skip_frac


def candidate_pool(history, seen, sasrec_i2i, lightfm_i2i):
    candidates = []
    added = set()

    anchors = history[-ANCHOR_WINDOW:]
    for track_id, _ in reversed(anchors):
        for cand in sasrec_i2i.get(track_id, [])[:TOPK_PER_SOURCE]:
            if cand not in seen and cand not in added:
                added.add(cand)
                candidates.append(cand)
                if len(candidates) >= MAX_CANDIDATES:
                    return candidates

        for cand in lightfm_i2i.get(track_id, [])[:TOPK_PER_SOURCE]:
            if cand not in seen and cand not in added:
                added.add(cand)
                candidates.append(cand)
                if len(candidates) >= MAX_CANDIDATES:
                    return candidates

    return candidates


def rank_features(neighbors, cand):
    if cand not in neighbors:
        return 0.0, 0.0
    rank = neighbors.index(cand) + 1
    return 1.0, 1.0 / rank


def build_feature_row(history, prev_track, prev_time, cand, tracks_meta, sasrec_i2i, lightfm_i2i):
    avg_time, last_time, good_frac, skip_frac = get_recent_stats(history)

    prev_meta = tracks_meta.get(prev_track, {})
    cand_meta = tracks_meta.get(cand, {})

    same_artist_last = 0.0
    same_title_prefix = 0.0
    cand_artist_repeat = 0.0

    prev_artist = prev_meta.get("artist")
    cand_artist = cand_meta.get("artist")

    if prev_artist is not None and cand_artist is not None:
        same_artist_last = 1.0 if prev_artist == cand_artist else 0.0

    prev_title = str(prev_meta.get("title", "") or "").lower()
    cand_title = str(cand_meta.get("title", "") or "").lower()
    if prev_title and cand_title:
        same_title_prefix = 1.0 if prev_title[:6] == cand_title[:6] else 0.0

    recent_tracks = [t for t, _ in history[-ANCHOR_WINDOW:]]
    recent_artists = [tracks_meta.get(t, {}).get("artist") for t in recent_tracks]
    cand_artist_repeat = float(sum(a == cand_artist for a in recent_artists if a is not None))

    sasrec_hits = 0.0
    lfm_hits = 0.0
    sasrec_rank_sum_inv = 0.0
    lfm_rank_sum_inv = 0.0
    source_agreement = 0.0

    for a_track, a_time in reversed(history[-ANCHOR_WINDOW:]):
        sas_neighbors = sasrec_i2i.get(a_track, [])[:TOPK_PER_SOURCE]
        lfm_neighbors = lightfm_i2i.get(a_track, [])[:TOPK_PER_SOURCE]

        sas_hit, sas_rr = rank_features(sas_neighbors, cand)
        lfm_hit, lfm_rr = rank_features(lfm_neighbors, cand)

        sasrec_hits += sas_hit
        lfm_hits += lfm_hit
        sasrec_rank_sum_inv += sas_rr * max(a_time, 0.05)
        lfm_rank_sum_inv += lfm_rr * max(a_time, 0.05)

        if sas_hit > 0 and lfm_hit > 0:
            source_agreement += 1.0

    return {
        "hist_len": float(len(history)),
        "recent_avg_time": avg_time,
        "recent_last_time": last_time,
        "recent_good_frac": good_frac,
        "recent_skip_frac": skip_frac,
        "same_artist_last": same_artist_last,
        "same_title_prefix": same_title_prefix,
        "cand_artist_repeat": cand_artist_repeat,
        "sasrec_hits": sasrec_hits,
        "lfm_hits": lfm_hits,
        "sasrec_rank_sum_inv": sasrec_rank_sum_inv,
        "lfm_rank_sum_inv": lfm_rank_sum_inv,
        "source_agreement": source_agreement,
    }


def main():
    np.random.seed(RANDOM_SEED)

    tracks_meta = load_tracks(TRACKS_PATH)
    sasrec_i2i = load_i2i(SASREC_PATH)
    lightfm_i2i = load_i2i(LIGHTFM_PATH)

    rows = read_control_logs(LOG_PATH)
    print("control rows:", len(rows))

    sessions = build_sessions(rows)
    print("sessions:", len(sessions))

    samples = []

    for user, session in sessions:
        # session 是按时间顺序的记录
        # 对每个状态，用当前 track 作为 prev，下一条记录的 track 作为真实 next
        for i in range(len(session) - 1):
            curr = session[i]
            nxt = session[i + 1]

            prev_track = int(curr["track"])
            prev_time = float(curr["time"])
            true_next = int(nxt["track"])
            true_next_time = float(nxt["time"])

            history = [(int(x["track"]), float(x["time"])) for x in session[: i + 1]]
            seen = {t for t, _ in history}

            cands = candidate_pool(history, seen, sasrec_i2i, lightfm_i2i)
            if true_next not in cands and true_next not in seen and true_next != prev_track:
                cands = [true_next] + cands

            cands = list(dict.fromkeys([c for c in cands if c != prev_track]))[:MAX_CANDIDATES]
            if len(cands) < 2:
                continue

            # 只保留高质量正样本状态
            if true_next_time >= GOOD_TIME:
                for cand in cands:
                    feat = build_feature_row(
                        history=history,
                        prev_track=prev_track,
                        prev_time=prev_time,
                        cand=cand,
                        tracks_meta=tracks_meta,
                        sasrec_i2i=sasrec_i2i,
                        lightfm_i2i=lightfm_i2i,
                    )
                    feat["user"] = user
                    feat["label"] = 1 if cand == true_next else 0
                    samples.append(feat)

    df = pd.DataFrame(samples)
    print("dataset shape:", df.shape)
    print(df["label"].value_counts())

    feature_cols = [
        "hist_len",
        "recent_avg_time",
        "recent_last_time",
        "recent_good_frac",
        "recent_skip_frac",
        "same_artist_last",
        "same_title_prefix",
        "cand_artist_repeat",
        "sasrec_hits",
        "lfm_hits",
        "sasrec_rank_sum_inv",
        "lfm_rank_sum_inv",
        "source_agreement",
    ]

    X = df[feature_cols]
    y = df["label"]
    groups = df["user"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, valid_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=400, class_weight="balanced", C=0.5)),
    ])

    pipe.fit(X_train, y_train)

    train_score = pipe.score(X_train, y_train)
    valid_score = pipe.score(X_valid, y_valid)

    print("train_acc:", round(float(train_score), 6))
    print("valid_acc:", round(float(valid_score), 6))

    bundle = {
        "model": pipe,
        "feature_cols": feature_cols,
        "params": {
            "good_time": GOOD_TIME,
            "skip_time": SKIP_TIME,
            "anchor_window": ANCHOR_WINDOW,
            "topk_per_source": TOPK_PER_SOURCE,
            "max_candidates": MAX_CANDIDATES,
        },
    }

    joblib.dump(bundle, OUTPUT_PATH)
    print("saved to", OUTPUT_PATH)


if __name__ == "__main__":
    main()