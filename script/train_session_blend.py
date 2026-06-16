import argparse
import glob
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_log_files(paths):
    resolved = []
    for path in paths:
        candidate = Path(path)
        if candidate.is_file():
            resolved.append(str(candidate))
        elif candidate.exists():
            resolved.extend(glob.glob(str(candidate / "**" / "data.json"), recursive=True))
        else:
            resolved.extend(glob.glob(path, recursive=True))
    return sorted(set(resolved))


def load_tracks(path):
    tracks = []
    artist_by_track = {}
    for row in read_jsonl(path):
        track = int(row["track"])
        tracks.append(track)
        artist_by_track[track] = row["artist"]
    return tracks, artist_by_track


def load_recommendations(path, key_object, key_recommendations):
    mapping = {}
    for row in read_jsonl(path):
        mapping[int(row[key_object])] = [int(track) for track in row[key_recommendations]]
    return mapping


def load_rows(paths, only_control):
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            if row.get("message") not in {"next", "last"}:
                continue
            if only_control and row.get("experiments", {}).get("HSTU") != "C":
                continue
            rows.append(row)
    rows.sort(key=lambda row: (int(row["user"]), int(row["timestamp"])))
    return rows


def build_stats(rows, artist_by_track):
    user_track_weight = defaultdict(float)
    user_seen = defaultdict(set)
    track_sum = defaultdict(float)
    track_count = defaultdict(float)
    artist_sum = defaultdict(float)
    artist_count = defaultdict(float)

    for row in rows:
        user = int(row["user"])
        track = int(row["track"])
        dwell = float(row["time"])
        user_track_weight[(user, track)] += np.log1p(max(dwell, 0.01) * 8.0)
        user_seen[user].add(track)
        track_sum[track] += dwell
        track_count[track] += 1.0
        artist = artist_by_track.get(track)
        if artist is not None:
            artist_sum[artist] += dwell
            artist_count[artist] += 1.0

    global_quality = float(sum(track_sum.values()) / max(sum(track_count.values()), 1.0))
    track_quality = {track: float(track_sum[track] / track_count[track]) for track in track_sum}
    artist_quality = {artist: float(artist_sum[artist] / artist_count[artist]) for artist in artist_sum}
    popularity_total = float(sum(track_count.values()) or 1.0)
    track_popularity = {track: float(track_count[track] / popularity_total) for track in track_count}

    return user_track_weight, user_seen, track_quality, artist_quality, track_popularity, global_quality


def fit_latent_candidates(user_track_weight, all_tracks, track_quality, user_seen, top_k, latent_dim, seed):
    users = sorted({user for user, _ in user_track_weight})
    user_to_index = {user: idx for idx, user in enumerate(users)}
    track_to_index = {track: idx for idx, track in enumerate(all_tracks)}

    rows = []
    cols = []
    data = []
    for (user, track), value in user_track_weight.items():
        if track not in track_to_index:
            continue
        rows.append(user_to_index[user])
        cols.append(track_to_index[track])
        data.append(float(value))
    matrix = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)),
        shape=(len(users), len(all_tracks)),
        dtype=np.float32,
    )
    svd = TruncatedSVD(n_components=latent_dim, random_state=seed)
    user_factors = normalize_rows(svd.fit_transform(matrix).astype(np.float32))
    item_factors = normalize_rows(svd.components_.T.astype(np.float32))
    all_tracks_array = np.asarray(all_tracks, dtype=np.int32)
    quality_vector = np.asarray([track_quality.get(track, 0.0) for track in all_tracks], dtype=np.float32)
    global_order = np.argsort(-(quality_vector))
    global_candidates = [int(all_tracks_array[idx]) for idx in global_order[: max(top_k, 200)]]

    result = {}
    for user, user_idx in user_to_index.items():
        scores = user_factors[user_idx] @ item_factors.T + 0.12 * quality_vector
        partial = np.argpartition(scores, -top_k * 3)[-top_k * 3 :]
        ranked = partial[np.argsort(-scores[partial])]
        seen = user_seen.get(user, set())
        picks = []
        for idx in ranked:
            track = int(all_tracks_array[idx])
            if track in seen:
                continue
            picks.append(track)
            if len(picks) >= top_k:
                break
        if len(picks) < top_k:
            for track in global_candidates:
                if track not in seen and track not in picks:
                    picks.append(track)
                if len(picks) >= top_k:
                    break
        result[int(user)] = picks
    return result, global_candidates


def normalize_rows(array):
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def time_bucket(dwell):
    if dwell >= 0.9:
        return "great"
    if dwell >= 0.65:
        return "good"
    if dwell >= 0.35:
        return "mid"
    return "bad"


def depth_bucket(length):
    if length >= 8:
        return "deep"
    if length >= 4:
        return "mid"
    return "short"


def anchors(prev_track, state_history):
    result = [("last", prev_track)]
    for track, dwell in sorted(state_history[:-1], key=lambda item: item[1], reverse=True):
        if dwell < 0.65:
            continue
        if all(existing != track for _, existing in result):
            result.append(("good", track))
        if len(result) >= 3:
            break
    return result


def rank_bucket(rank):
    if rank <= 3:
        return "top3"
    if rank <= 10:
        return "top10"
    if rank <= 30:
        return "top30"
    return "tail"


def contains_with_rank(sequence, target, limit):
    for index, track in enumerate(sequence[:limit], start=1):
        if int(track) == target:
            return index
    return None


def train_blend(
    rows,
    artist_by_track,
    latent_candidates,
    hstu_candidates,
    sasrec_candidates,
    lightfm_candidates,
    source_limits,
):
    by_user = defaultdict(list)
    for row in rows:
        by_user[int(row["user"])].append(row)

    context_sum = defaultdict(float)
    context_count = defaultdict(float)
    rank_sum = defaultdict(float)
    rank_count = defaultdict(float)
    source_sum = defaultdict(float)
    source_count = defaultdict(float)

    for user, user_rows in by_user.items():
        future_session_value = [0.0] * len(user_rows)
        running = 0.0
        for rev_idx in range(len(user_rows) - 1, -1, -1):
            running += float(user_rows[rev_idx]["time"])
            future_session_value[rev_idx] = running
            if user_rows[rev_idx].get("message") == "last":
                running = 0.0

        history = []
        for idx, row in enumerate(user_rows[:-1]):
            track = int(row["track"])
            dwell = float(row["time"])
            if row.get("message") != "next" or row.get("recommendation") is None:
                history.append((track, dwell))
                history = history[-10:]
                continue

            next_row = user_rows[idx + 1]
            recommendation = int(row["recommendation"])
            if int(next_row["track"]) != recommendation:
                history.append((track, dwell))
                history = history[-10:]
                continue

            state_history = (history + [(track, dwell)])[-10:]
            context_key = f"{time_bucket(dwell)}|{depth_bucket(len(state_history))}"
            value = float(future_session_value[idx + 1])
            matched = False

            latent_rank = contains_with_rank(latent_candidates.get(user, ()), recommendation, source_limits["latent"])
            if latent_rank is not None:
                matched = True
                context_sum[(context_key, "latent|user")] += value
                context_count[(context_key, "latent|user")] += 1.0
                rank_sum[("latent", rank_bucket(latent_rank))] += value
                rank_count[("latent", rank_bucket(latent_rank))] += 1.0
                source_sum["latent"] += value
                source_count["latent"] += 1.0

            hstu_rank = contains_with_rank(hstu_candidates.get(user, ()), recommendation, source_limits["hstu"])
            if hstu_rank is not None:
                matched = True
                context_sum[(context_key, "hstu|user")] += value
                context_count[(context_key, "hstu|user")] += 1.0
                rank_sum[("hstu", rank_bucket(hstu_rank))] += value
                rank_count[("hstu", rank_bucket(hstu_rank))] += 1.0
                source_sum["hstu"] += value
                source_count["hstu"] += 1.0

            for anchor_kind, anchor_track in anchors(track, state_history):
                sasrec_rank = contains_with_rank(sasrec_candidates.get(anchor_track, ()), recommendation, source_limits["sasrec"])
                if sasrec_rank is not None:
                    matched = True
                    context_sum[(context_key, f"sasrec|{anchor_kind}")] += value
                    context_count[(context_key, f"sasrec|{anchor_kind}")] += 1.0
                    rank_sum[("sasrec", rank_bucket(sasrec_rank))] += value
                    rank_count[("sasrec", rank_bucket(sasrec_rank))] += 1.0
                    source_sum["sasrec"] += value
                    source_count["sasrec"] += 1.0
                lightfm_rank = contains_with_rank(lightfm_candidates.get(anchor_track, ()), recommendation, source_limits["lightfm"])
                if lightfm_rank is not None:
                    matched = True
                    context_sum[(context_key, f"lightfm|{anchor_kind}")] += value
                    context_count[(context_key, f"lightfm|{anchor_kind}")] += 1.0
                    rank_sum[("lightfm", rank_bucket(lightfm_rank))] += value
                    rank_count[("lightfm", rank_bucket(lightfm_rank))] += 1.0
                    source_sum["lightfm"] += value
                    source_count["lightfm"] += 1.0

            if not matched:
                source_sum["fallback"] += value
                source_count["fallback"] += 1.0

            history.append((track, dwell))
            history = history[-10:]

    global_mean = float(sum(source_sum.values()) / max(sum(source_count.values()), 1.0))

    context_weights = defaultdict(dict)
    for (context_key, feature_key), total in context_sum.items():
        count = context_count[(context_key, feature_key)]
        context_weights[context_key][feature_key] = float((total + global_mean * 8.0) / (count + 8.0) - global_mean)

    rank_weights = defaultdict(dict)
    for (source, bucket), total in rank_sum.items():
        count = rank_count[(source, bucket)]
        rank_weights[source][bucket] = float((total + global_mean * 10.0) / (count + 10.0) - global_mean)

    source_bias = {}
    for source, total in source_sum.items():
        count = source_count[source]
        source_bias[source] = float((total + global_mean * 15.0) / (count + 15.0) - global_mean)

    return context_weights, rank_weights, source_bias, global_mean


def save_artifact(
    output_path,
    latent_candidates,
    global_candidates,
    track_quality,
    artist_quality,
    track_popularity,
    global_quality,
    context_weights,
    rank_weights,
    source_bias,
    history_limit,
    source_limits,
):
    payload = {
        "latent_candidates": latent_candidates,
        "global_candidates": global_candidates,
        "track_quality": track_quality,
        "artist_quality": artist_quality,
        "track_popularity": track_popularity,
        "global_quality": global_quality,
        "context_weights": dict(context_weights),
        "rank_weights": {source: dict(values) for source, values in rank_weights.items()},
        "source_bias": source_bias,
        "history_limit": history_limit,
        "source_limits": source_limits,
        "min_score": global_quality * 0.55,
    }
    Path(output_path).write_bytes(pickle.dumps(payload, protocol=4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--tracks", default="botify/data/tracks.json")
    parser.add_argument("--sasrec", default="botify/data/sasrec_i2i.jsonl")
    parser.add_argument("--lightfm", default="botify/data/lightfm_i2i.jsonl")
    parser.add_argument("--hstu", default="botify/data/hstu_recommendations.json")
    parser.add_argument("--output", default="botify/data/session_blend_model.pkl")
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=120)
    parser.add_argument("--seed", type=int, default=31312)
    parser.add_argument("--include-all", action="store_true")
    args = parser.parse_args()

    log_files = find_log_files(args.logs)
    if not log_files:
        raise RuntimeError("log files not found")

    all_tracks, artist_by_track = load_tracks(args.tracks)
    sasrec_candidates = load_recommendations(args.sasrec, "item_id", "recommendations")
    lightfm_candidates = load_recommendations(args.lightfm, "item_id", "recommendations")
    hstu_candidates = load_recommendations(args.hstu, "user", "tracks")
    rows = load_rows(log_files, only_control=not args.include_all)

    user_track_weight, user_seen, track_quality, artist_quality, track_popularity, global_quality = build_stats(rows, artist_by_track)
    latent_candidates, global_candidates = fit_latent_candidates(
        user_track_weight=user_track_weight,
        all_tracks=all_tracks,
        track_quality=track_quality,
        user_seen=user_seen,
        top_k=args.top_k,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )
    source_limits = {
        "latent": 50,
        "hstu": 50,
        "sasrec": 30,
        "lightfm": 30,
        "global": 60,
    }
    context_weights, rank_weights, source_bias, global_mean = train_blend(
        rows=rows,
        artist_by_track=artist_by_track,
        latent_candidates=latent_candidates,
        hstu_candidates=hstu_candidates,
        sasrec_candidates=sasrec_candidates,
        lightfm_candidates=lightfm_candidates,
        source_limits=source_limits,
    )
    save_artifact(
        output_path=args.output,
        latent_candidates=latent_candidates,
        global_candidates=global_candidates,
        track_quality=track_quality,
        artist_quality=artist_quality,
        track_popularity=track_popularity,
        global_quality=global_mean,
        context_weights=context_weights,
        rank_weights=rank_weights,
        source_bias=source_bias,
        history_limit=10,
        source_limits=source_limits,
    )
    print(f"saved artifact to {args.output}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
