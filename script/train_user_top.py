import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_log_files(paths):
    result = []
    for path in paths:
        p = Path(path)
        if p.is_file():
            result.append(str(p))
        elif p.exists():
            result.extend(glob.glob(str(p / "**" / "data.json"), recursive=True))
        else:
            result.extend(glob.glob(path, recursive=True))
    return sorted(set(result))


def load_tracks(path):
    tracks = []
    artists = {}
    for row in read_jsonl(path):
        track = int(row["track"])
        tracks.append(track)
        artists[track] = row["artist"]
    return sorted(tracks), artists


def load_old_recs(path):
    result = {}
    if not path:
        return result
    for row in read_jsonl(path):
        result[int(row["user"])] = [int(x) for x in row["tracks"]]
    return result


def load_i2i(path):
    result = {}
    for row in read_jsonl(path):
        result[int(row["item_id"])] = [int(x) for x in row["recommendations"]]
    return result


def load_events(paths, min_time, only_control):
    user_track_score = defaultdict(float)
    user_seen = defaultdict(set)
    users = set()
    item_score = Counter()

    for path in paths:
        for row in read_jsonl(path):
            if row.get("message") not in ("next", "last"):
                continue
            if only_control and row.get("experiments", {}).get("HSTU") != "C":
                continue
            user = int(row["user"])
            track = int(row["track"])
            t = float(row["time"])
            users.add(user)
            user_seen[user].add(track)
            if t >= min_time:
                user_track_score[(user, track)] += t
                item_score[track] += 1

    return user_track_score, user_seen, sorted(users), item_score


def load_rows(paths, only_control):
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            if row.get("message") not in ("next", "last"):
                continue
            if only_control and row.get("experiments", {}).get("HSTU") != "C":
                continue
            rows.append(row)
    return sorted(rows, key=lambda x: x["timestamp"])


def recommend(user_track_score, user_seen, users, fallback, popular, sasrec, lightfm, top_k):
    by_user = defaultdict(list)
    for (user, track), score in user_track_score.items():
        by_user[user].append((track, score))
    result = {}

    for user in users:
        scores = Counter()
        positives = sorted(by_user[user], key=lambda x: x[1], reverse=True)
        seen = user_seen[user]

        for track, score in positives[:20]:
            scores[track] += 0.25 * score
            for i, candidate in enumerate(sasrec.get(track, ())):
                scores[candidate] += score * (1.0 / (i + 2))
            for i, candidate in enumerate(lightfm.get(track, ())):
                scores[candidate] += score * (0.7 / (i + 2))

        recs = []
        for track, _ in scores.most_common():
            if track in seen:
                continue
            recs.append(int(track))
            if len(recs) >= top_k:
                break

        if len(recs) < top_k:
            for track in fallback.get(user, popular):
                if track not in seen and track not in recs:
                    recs.append(int(track))
                if len(recs) >= top_k:
                    break

        if len(recs) < top_k:
            for track in popular:
                if track not in seen and track not in recs:
                    recs.append(int(track))
                if len(recs) >= top_k:
                    break

        result[int(user)] = recs[:top_k]

    for user, recs in fallback.items():
        result.setdefault(int(user), [int(x) for x in recs[:top_k]])

    return result


def time_bucket(value):
    if value < 0.35:
        return "bad"
    if value < 0.65:
        return "mid"
    if value < 0.9:
        return "good"
    return "great"


def rank_bucket(rank):
    if rank <= 2:
        return "top3"
    if rank <= 9:
        return "top10"
    if rank <= 29:
        return "top30"
    return "none"


def anchor_bucket(pos):
    if pos == 0:
        return "last"
    if pos <= 3:
        return "session"
    return "none"


def artist_bucket(count):
    if count == 0:
        return "new"
    if count == 1:
        return "once"
    return "repeat"


def mean_bucket(mean):
    if mean < 0.35:
        return "low"
    if mean < 0.55:
        return "mid"
    if mean < 0.75:
        return "high"
    return "very_high"


def session_candidates(user, history, prev_track, sasrec, lightfm, user_recs):
    anchors = [prev_track]
    for track, score in sorted(history, key=lambda x: x[1], reverse=True):
        if score >= 0.75 and track not in anchors:
            anchors.append(track)
        if len(anchors) >= 4:
            break

    result = {}
    for anchor_pos, anchor in enumerate(anchors):
        for name, model in (("sasrec", sasrec), ("lightfm", lightfm)):
            for rank, track in enumerate(model.get(anchor, ())[:30]):
                info = result.setdefault(track, {"sources": set(), "best_rank": 1000, "anchor_pos": 1000, "user_rank": 1000})
                info["sources"].add(name)
                info["best_rank"] = min(info["best_rank"], rank)
                info["anchor_pos"] = min(info["anchor_pos"], anchor_pos)

    for rank, track in enumerate(user_recs.get(user, ())[:50]):
        info = result.setdefault(track, {"sources": set(), "best_rank": 1000, "anchor_pos": 1000, "user_rank": 1000})
        info["sources"].add("user")
        info["user_rank"] = min(info["user_rank"], rank)

    return result


def make_feature_keys(track, info, prev_time, history, artists, track_mean, artist_mean, global_mean):
    seen_artists = Counter()
    for old_track, _ in history:
        artist = artists.get(old_track)
        if artist is not None:
            seen_artists[artist] += 1

    artist = artists.get(track)
    source_key = "+".join(sorted(info["sources"]))
    return [
        "prev_time=" + time_bucket(prev_time),
        "source=" + source_key,
        "rank=" + rank_bucket(info["best_rank"]),
        "anchor=" + anchor_bucket(info["anchor_pos"]),
        "user_rank=" + rank_bucket(info["user_rank"]),
        "artist_seen=" + artist_bucket(seen_artists.get(artist, 0)),
        "track_prior=" + mean_bucket(track_mean.get(track, global_mean)),
        "artist_prior=" + mean_bucket(artist_mean.get(artist, global_mean)),
    ]


def train_value_model(rows, user_recs, sasrec, lightfm, artists, max_negatives):
    by_user = defaultdict(list)
    track_sum = defaultdict(float)
    track_count = defaultdict(float)
    artist_sum = defaultdict(float)
    artist_count = defaultdict(float)

    for row in rows:
        by_user[int(row["user"])].append(row)
        track = int(row["track"])
        value = float(row["time"])
        track_sum[track] += value
        track_count[track] += 1.0
        artist = artists.get(track)
        if artist is not None:
            artist_sum[artist] += value
            artist_count[artist] += 1.0

    global_sum = sum(track_sum.values())
    global_count = sum(track_count.values()) or 1.0
    global_mean = global_sum / global_count
    track_mean = {track: track_sum[track] / track_count[track] for track in track_sum}
    artist_mean = {artist: artist_sum[artist] / artist_count[artist] for artist in artist_sum}

    value_sum = defaultdict(float)
    value_count = defaultdict(float)

    def add(keys, y, weight):
        for key in keys:
            value_sum[key] += y * weight
            value_count[key] += weight

    for user, user_rows in by_user.items():
        history = []
        for i, row in enumerate(user_rows[:-1]):
            if row.get("message") != "next" or row.get("recommendation") is None:
                history.append((int(row["track"]), float(row["time"])))
                history = history[-10:]
                continue

            prev_track = int(row["track"])
            prev_time = float(row["time"])
            recommendation = int(row["recommendation"])
            next_row = user_rows[i + 1]
            if int(next_row["track"]) != recommendation:
                history.append((prev_track, prev_time))
                history = history[-10:]
                continue

            state_history = (history + [(prev_track, prev_time)])[-10:]
            candidates = session_candidates(user, state_history, prev_track, sasrec, lightfm, user_recs)
            if recommendation not in candidates:
                candidates[recommendation] = {"sources": {"shown"}, "best_rank": 1000, "anchor_pos": 1000, "user_rank": 1000}

            y = float(next_row["time"])
            add(make_feature_keys(recommendation, candidates[recommendation], prev_time, state_history, artists, track_mean, artist_mean, global_mean), y, 2.0)

            negatives = 0
            for candidate, info in candidates.items():
                if candidate == recommendation:
                    continue
                add(make_feature_keys(candidate, info, prev_time, state_history, artists, track_mean, artist_mean, global_mean), 0.05, 0.15)
                negatives += 1
                if negatives >= max_negatives:
                    break

            history.append((prev_track, prev_time))
            history = history[-10:]

    values = {}
    for key in value_sum:
        count = value_count[key]
        mean = (value_sum[key] + global_mean * 10.0) / (count + 10.0)
        values[key] = {"mean": round(mean, 6), "count": round(count, 3)}

    return {
        "global_mean": round(global_mean, 6),
        "min_score": 0.12,
        "values": values,
        "track_mean": {str(k): round(v, 6) for k, v in track_mean.items() if track_count[k] >= 3},
        "artist_mean": {k: round(v, 6) for k, v in artist_mean.items() if artist_count[k] >= 3},
    }


def save_recs(recs, output):
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for user in sorted(recs):
            f.write(json.dumps({"user": user, "tracks": recs[user]}, ensure_ascii=False) + "\n")


def save_model(model, output):
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--tracks", default="botify/data/tracks.json")
    parser.add_argument("--fallback", default="botify/data/hstu_recommendations.json")
    parser.add_argument("--sasrec", default="botify/data/sasrec_i2i.jsonl")
    parser.add_argument("--lightfm", default="botify/data/lightfm_i2i.jsonl")
    parser.add_argument("--output", default="botify/data/user_top_recommendations.json")
    parser.add_argument("--model-output", default="botify/data/session_value_model.json")
    parser.add_argument("--min-time", type=float, default=0.72)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--only-control", action="store_true")
    parser.add_argument("--max-negatives", type=int, default=20)
    args = parser.parse_args()

    log_files = find_log_files(args.logs)
    if not log_files:
        raise RuntimeError("log files not found")

    user_track_score, user_seen, users, item_score = load_events(log_files, args.min_time, args.only_control)
    if not user_track_score:
        raise RuntimeError("no positive events")

    fallback = load_old_recs(args.fallback)
    all_items, artists = load_tracks(args.tracks)
    sasrec = load_i2i(args.sasrec)
    lightfm = load_i2i(args.lightfm)
    popular = [track for track, _ in item_score.most_common()]
    if popular:
        seen_popular = set(popular)
        popular = popular + [x for x in all_items if x not in seen_popular]
    else:
        popular = all_items

    recs = recommend(user_track_score, user_seen, users, fallback, popular, sasrec, lightfm, args.top_k)
    save_recs(recs, args.output)
    rows = load_rows(log_files, args.only_control)
    model = train_value_model(rows, recs, sasrec, lightfm, artists, args.max_negatives)
    save_model(model, args.model_output)
    print(f"saved {len(recs)} users to {args.output}")
    print(f"saved value model to {args.model_output}")


if __name__ == "__main__":
    main()
