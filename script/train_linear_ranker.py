import argparse
import collections
import json
import math
import random
from pathlib import Path

from sklearn.linear_model import LogisticRegression


def load_tracks(path):
    tracks = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tracks[int(row["track"])] = row
    return tracks


def load_i2i(path, key_object="item_id", key_recommendations="recommendations"):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            data[int(row[key_object])] = [int(x) for x in row[key_recommendations]]
    return data


def load_logs(log_paths):
    rows = []
    for path in log_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("message") == "next" and row.get("recommendation") is not None:
                    rows.append(row)
    rows.sort(key=lambda x: (int(x["user"]), int(x["timestamp"])))
    return rows


def build_user_histories(events):
    grouped = collections.defaultdict(list)
    for row in events:
        grouped[int(row["user"])].append((int(row["track"]), float(row["time"]), int(row["recommendation"])))
    return grouped


def recency_anchor_weights(history, max_recent_anchors):
    total_time = collections.defaultdict(float)
    for track, listened_time in history:
        total_time[int(track)] += float(listened_time)

    anchors = []
    used = set()
    unique_idx = 0
    for track, _ in history:
        track = int(track)
        if track in used:
            continue
        used.add(track)
        recency_weight = 1.0 / float(unique_idx + 1)
        strength = math.log1p(total_time[track])
        anchors.append(
            {
                "track": track,
                "weight": strength * recency_weight,
                "is_last": 1 if unique_idx == 0 else 0,
                "is_second": 1 if unique_idx == 1 else 0,
            }
        )
        unique_idx += 1
        if unique_idx >= max_recent_anchors:
            break
    return anchors


def candidate_features(candidate, user_history, anchors, sasrec, lightfm, hstu, track_artist, max_sasrec, max_lightfm, max_hstu):
    seen_tracks = set(track for track, _ in user_history)
    if candidate in seen_tracks:
        return None

    artist_counts = collections.Counter(track_artist.get(track, "__unknown_artist__") for track, _ in user_history)
    last_track = int(user_history[0][0])
    last_artist = track_artist.get(last_track)

    feat = {
        "in_sasrec": 0.0,
        "in_lightfm": 0.0,
        "in_hstu": 0.0,
        "sasrec_best_rr": 0.0,
        "lightfm_best_rr": 0.0,
        "hstu_best_rr": 0.0,
        "sasrec_last_rr": 0.0,
        "lightfm_last_rr": 0.0,
        "sasrec_second_rr": 0.0,
        "lightfm_second_rr": 0.0,
        "source_votes": 0.0,
        "consensus": 0.0,
        "same_artist_as_last": 0.0,
        "artist_seen_count": 0.0,
        "fresh_artist": 0.0,
        "anchor_weight_sum": 0.0,
        "anchor_weight_max": 0.0,
        "anchor_count": 0.0,
    }

    touched_by_anchor = set()
    for anchor in anchors:
        s_list = sasrec.get(anchor["track"], [])[:max_sasrec]
        l_list = lightfm.get(anchor["track"], [])[:max_lightfm]

        if candidate in s_list:
            rank = s_list.index(candidate) + 1
            rr = 1.0 / float(rank)
            feat["in_sasrec"] = 1.0
            feat["sasrec_best_rr"] = max(feat["sasrec_best_rr"], rr)
            if anchor["is_last"]:
                feat["sasrec_last_rr"] = max(feat["sasrec_last_rr"], rr)
            if anchor["is_second"]:
                feat["sasrec_second_rr"] = max(feat["sasrec_second_rr"], rr)
            feat["anchor_weight_sum"] += anchor["weight"]
            feat["anchor_weight_max"] = max(feat["anchor_weight_max"], anchor["weight"])
            touched_by_anchor.add(anchor["track"])

        if candidate in l_list:
            rank = l_list.index(candidate) + 1
            rr = 1.0 / float(rank)
            feat["in_lightfm"] = 1.0
            feat["lightfm_best_rr"] = max(feat["lightfm_best_rr"], rr)
            if anchor["is_last"]:
                feat["lightfm_last_rr"] = max(feat["lightfm_last_rr"], rr)
            if anchor["is_second"]:
                feat["lightfm_second_rr"] = max(feat["lightfm_second_rr"], rr)
            feat["anchor_weight_sum"] += anchor["weight"]
            feat["anchor_weight_max"] = max(feat["anchor_weight_max"], anchor["weight"])
            touched_by_anchor.add(anchor["track"])

    if candidate in hstu[:max_hstu]:
        rank = hstu.index(candidate) + 1
        feat["in_hstu"] = 1.0
        feat["hstu_best_rr"] = max(feat["hstu_best_rr"], 1.0 / float(rank))

    feat["source_votes"] = feat["in_sasrec"] + feat["in_lightfm"] + feat["in_hstu"]
    feat["consensus"] = 1.0 if feat["in_sasrec"] and feat["in_lightfm"] else 0.0
    feat["anchor_count"] = float(len(touched_by_anchor))

    candidate_artist = track_artist.get(candidate)
    if candidate_artist == last_artist:
        feat["same_artist_as_last"] = 1.0
    feat["artist_seen_count"] = float(artist_counts.get(candidate_artist, 0))
    feat["fresh_artist"] = 1.0 if feat["artist_seen_count"] == 0 else 0.0

    return feat


def rows_to_matrix(rows, feature_order):
    return [[row.get(name, 0.0) for name in feature_order] for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--sasrec", required=True)
    parser.add_argument("--lightfm", required=True)
    parser.add_argument("--hstu", required=True)
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=31312)
    parser.add_argument("--negatives-per-positive", type=int, default=6)
    args = parser.parse_args()

    random.seed(args.seed)

    tracks = load_tracks(args.tracks)
    sasrec = load_i2i(args.sasrec)
    lightfm = load_i2i(args.lightfm)
    hstu = load_i2i(args.hstu, key_object="user", key_recommendations="tracks")

    track_artist = {track_id: row["artist"] for track_id, row in tracks.items()}

    events = load_logs(args.logs)
    user_histories = build_user_histories(events)

    feature_order = [
        "in_sasrec",
        "in_lightfm",
        "in_hstu",
        "sasrec_best_rr",
        "lightfm_best_rr",
        "hstu_best_rr",
        "sasrec_last_rr",
        "lightfm_last_rr",
        "sasrec_second_rr",
        "lightfm_second_rr",
        "source_votes",
        "consensus",
        "same_artist_as_last",
        "artist_seen_count",
        "fresh_artist",
        "anchor_weight_sum",
        "anchor_weight_max",
        "anchor_count",
    ]

    X_rows = []
    y = []

    for user, history_rows in user_histories.items():
        history = []
        for track, listened_time, chosen in history_rows:
            history.insert(0, (track, listened_time))
            anchors = recency_anchor_weights(history, max_recent_anchors=3)
            seen = set(t for t, _ in history)

            candidates = set()
            for anchor in anchors:
                candidates.update(sasrec.get(anchor["track"], [])[:10])
                candidates.update(lightfm.get(anchor["track"], [])[:10])
            candidates.update(hstu.get(user, [])[:50])
            candidates = [c for c in candidates if c not in seen]

            if chosen not in candidates:
                candidates.append(chosen)

            pos_feat = candidate_features(
                chosen, history, anchors, sasrec, lightfm, hstu.get(user, []),
                track_artist, 10, 10, 50
            )
            if pos_feat is None:
                continue

            X_rows.append(pos_feat)
            y.append(1)

            negatives = [c for c in candidates if c != chosen]
            random.shuffle(negatives)
            for negative in negatives[: args.negatives_per_positive]:
                neg_feat = candidate_features(
                    negative, history, anchors, sasrec, lightfm, hstu.get(user, []),
                    track_artist, 10, 10, 50
                )
                if neg_feat is None:
                    continue
                X_rows.append(neg_feat)
                y.append(0)

        # history is not shared between users

    if not X_rows:
        raise RuntimeError("No training rows were created from the provided logs.")

    X = rows_to_matrix(X_rows, feature_order)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
    clf.fit(X, y)

    weights = {"bias": float(clf.intercept_[0])}
    for name, value in zip(feature_order, clf.coef_[0]):
        weights[name] = float(value)
    weights["max_recent_anchors"] = 3
    weights["max_candidates_per_source"] = {"sasrec": 10, "lightfm": 10, "hstu": 50}

    output_path = Path(args.output)
    output_path.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", output_path)


if __name__ == "__main__":
    main()
