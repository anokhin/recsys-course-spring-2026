import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split


FEATURE_NAMES = [
    "prev_track_time",
    "hist_len",
    "hist_avg_time",
    "hist_last_time",
    "hist_good_frac",
    "hist_skip_frac",

    "sasrec_hit",
    "lfm_hit",
    "hstu_hit",
    "same_artist_hit",
    "source_count",

    "sasrec_rank_inv",
    "lfm_rank_inv",
    "hstu_rank_inv",
    "same_artist_rank_inv",

    "sasrec_best_rank_norm",
    "lfm_best_rank_norm",
    "hstu_best_rank_norm",

    "same_artist_prev",
    "artist_recent_count",
    "candidate_in_recent_history",

    "diff_source_count_vs_baseline",
    "diff_total_rank_inv_vs_baseline",
    "diff_hstu_rank_inv_vs_baseline",
]


def _to_int(x, default=None):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def load_json_or_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        return []

    text = p.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    # Try normal JSON first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback to JSONL.
    rows = []
    with p.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_i2i_jsonl(path: str) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    if not path or not Path(path).exists():
        return out

    data = load_json_or_jsonl(path)

    if isinstance(data, dict):
        iterable = data.values() if "item_id" not in data else [data]
    else:
        iterable = data

    for obj in iterable:
        if not isinstance(obj, dict):
            continue
        key = _to_int(obj.get("item_id") or obj.get("track_id") or obj.get("track") or obj.get("id"))
        recs = obj.get("recommendations") or obj.get("tracks") or obj.get("items") or []
        if key is not None:
            out[key] = [_to_int(x) for x in recs if _to_int(x) is not None]
    return out


def load_user_hstu(path: str) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    if not path or not Path(path).exists():
        return out

    data = load_json_or_jsonl(path)

    if isinstance(data, dict):
        # Supports either {"123": [...]} or {"user": 123, "tracks": [...]}.
        if "user" in data or "user_id" in data:
            iterable = [data]
        else:
            for user, tracks in data.items():
                uid = _to_int(user)
                if uid is not None and isinstance(tracks, list):
                    out[uid] = [_to_int(x) for x in tracks if _to_int(x) is not None]
            return out
    else:
        iterable = data

    for obj in iterable:
        if not isinstance(obj, dict):
            continue
        user = _to_int(obj.get("user") or obj.get("user_id") or obj.get("id"))
        tracks = obj.get("tracks") or obj.get("recommendations") or obj.get("items") or []
        if user is not None:
            out[user] = [_to_int(x) for x in tracks if _to_int(x) is not None]
    return out


def load_tracks(path: str):
    track_to_artist: Dict[int, str] = {}
    if not path or not Path(path).exists():
        return track_to_artist, defaultdict(list)

    data = load_json_or_jsonl(path)

    if isinstance(data, dict):
        if "tracks" in data and isinstance(data["tracks"], list):
            iterable = data["tracks"]
        else:
            iterable = data.values()
    else:
        iterable = data

    for obj in iterable:
        if not isinstance(obj, dict):
            continue

        tid = _to_int(obj.get("track") or obj.get("track_id") or obj.get("item_id") or obj.get("id"))

        artist = (
            obj.get("artist")
            or obj.get("artist_id")
            or obj.get("artists")
            or obj.get("artist_name")
        )

        if isinstance(artist, dict):
            artist = artist.get("id") or artist.get("artist_id") or artist.get("name")

        if isinstance(artist, list):
            if len(artist) == 0:
                artist = None
            elif isinstance(artist[0], dict):
                artist = artist[0].get("id") or artist[0].get("artist_id") or artist[0].get("name")
            else:
                artist = artist[0]

        if tid is not None and artist is not None:
            track_to_artist[tid] = str(artist)

    artist_to_tracks = defaultdict(list)
    for tid, artist in track_to_artist.items():
        artist_to_tracks[artist].append(tid)

    return track_to_artist, artist_to_tracks


def iter_log_events(log_dir: str):
    p = Path(log_dir)
    if p.is_file():
        paths = [p]
    else:
        paths = sorted(set(list(p.rglob("*.json")) + list(p.rglob("*.jsonl"))))

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def get_group(event: dict) -> Optional[str]:
    exp = event.get("experiments") or event.get("experiment") or {}
    if isinstance(exp, dict):
        if "HSTU" in exp:
            return exp.get("HSTU")
        if exp:
            return next(iter(exp.values()))
    return event.get("group") or event.get("treatment")


def extract_event(event: dict):
    user = _to_int(event.get("user") or event.get("user_id"))
    consumed_track = _to_int(event.get("track") or event.get("track_id") or event.get("item_id"))
    recommendation = _to_int(event.get("recommendation") or event.get("recommended") or event.get("recommended_track"))

    if recommendation is None:
        recommendation = consumed_track

    t = event.get("time") or event.get("listen_time") or event.get("reward") or 0.0
    try:
        t = float(t)
    except Exception:
        t = 0.0

    return user, consumed_track, recommendation, t


def add_source_candidates(source_info, candidates, recs, source_name, top_k, weight=1.0, anchor=None):
    for rank, cand in enumerate(recs[:top_k], start=1):
        cand = _to_int(cand)
        if cand is None:
            continue
        if anchor is not None and cand == anchor:
            continue
        candidates.add(cand)
        source_info[cand][f"{source_name}_hit"] += 1.0
        source_info[cand][f"{source_name}_rank_inv"] += float(weight) / float(rank)
        key = f"{source_name}_best_rank"
        source_info[cand][key] = min(float(source_info[cand].get(key, 1e9)), float(rank))
        source_info[cand]["source_count"] += 1.0


def add_same_artist_candidates(
    source_info,
    candidates,
    prev_track: Optional[int],
    track_to_artist: Dict[int, str],
    artist_to_tracks: Dict[str, List[int]],
    top_k: int,
):
    if prev_track is None:
        return
    artist = track_to_artist.get(prev_track)
    if artist is None:
        return

    recs = [t for t in artist_to_tracks.get(artist, []) if int(t) != int(prev_track)]
    recs = recs[:top_k]
    add_source_candidates(source_info, candidates, recs, "same_artist", top_k, weight=1.0, anchor=prev_track)


def build_features(
    cand: int,
    baseline_item: int,
    prev_track: Optional[int],
    prev_time: float,
    user_history: List[Tuple[int, float]],
    source_info: Dict[int, Dict[str, float]],
    baseline_source: Dict[str, float],
    track_to_artist: Dict[int, str],
    history_limit: int,
):
    info = source_info.get(cand, {})

    hist_len = len(user_history)
    # Use only the most recent `history_limit` entries
    recent_history = user_history[-history_limit:] if history_limit > 0 else []
    times = [float(x[1]) for x in recent_history]
    hist_avg = float(np.mean(times)) if times else 0.0
    hist_last = float(times[-1]) if times else 0.0
    hist_good_frac = float(np.mean([x >= 0.75 for x in times])) if times else 0.0
    hist_skip_frac = float(np.mean([x <= 0.30 for x in times])) if times else 0.0

    cand_artist = track_to_artist.get(cand)
    prev_artist = track_to_artist.get(prev_track) if prev_track is not None else None
    recent_artists = [track_to_artist.get(t) for t, _ in recent_history]

    same_artist_prev = 1.0 if cand_artist is not None and cand_artist == prev_artist else 0.0
    artist_recent_count = float(sum(1 for a in recent_artists if a is not None and a == cand_artist))
    candidate_in_recent_history = 1.0 if cand in {t for t, _ in recent_history} else 0.0

    artist_recent_count_norm = artist_recent_count / 10.0

    sas_inv = float(info.get("sasrec_rank_inv", 0.0))
    lfm_inv = float(info.get("lfm_rank_inv", 0.0))
    hstu_inv = float(info.get("hstu_rank_inv", 0.0))
    same_inv = float(info.get("same_artist_rank_inv", 0.0))
    total_inv = sas_inv + lfm_inv + hstu_inv + same_inv

    base_total_inv = (
        float(baseline_source.get("sasrec_rank_inv", 0.0))
        + float(baseline_source.get("lfm_rank_inv", 0.0))
        + float(baseline_source.get("hstu_rank_inv", 0.0))
        + float(baseline_source.get("same_artist_rank_inv", 0.0))
    )

    def rank_norm(name):
        r = float(info.get(f"{name}_best_rank", 1e9))
        return 0.0 if r >= 1e8 else 1.0 / (1.0 + r)

    x = [
        float(prev_time or 0.0),
        float(min(hist_len, 50)) / 50.0,
        hist_avg,
        hist_last,
        hist_good_frac,
        hist_skip_frac,

        float(info.get("sasrec_hit", 0.0) > 0),
        float(info.get("lfm_hit", 0.0) > 0),
        float(info.get("hstu_hit", 0.0) > 0),
        float(info.get("same_artist_hit", 0.0) > 0),
        float(info.get("source_count", 0.0)) / 4.0,

        sas_inv,
        lfm_inv,
        hstu_inv,
        same_inv,

        rank_norm("sasrec"),
        rank_norm("lfm"),
        rank_norm("hstu"),

        same_artist_prev,
        artist_recent_count_norm,
        candidate_in_recent_history,

        float(info.get("source_count", 0.0)) - float(baseline_source.get("source_count", 0.0)),
        total_inv - base_total_inv,
        hstu_inv - float(baseline_source.get("hstu_rank_inv", 0.0)),
    ]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--tracks-path", default="./data/tracks.json")
    ap.add_argument("--sasrec-path", default="./data/sasrec_i2i.jsonl")
    ap.add_argument("--hstu-path", default="./data/hstu_recommendations.json")
    ap.add_argument("--lfm-path", default="./data/lightfm_i2i.jsonl")
    ap.add_argument("--use-only-control", action="store_true")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--history-limit", type=int, default=10,
                    help="Number of recent user interactions to use for features (must match online service).")
    ap.add_argument("--max-negatives-per-event", type=int, default=10)
    ap.add_argument("--positive-threshold", type=float, default=0.75)
    ap.add_argument("--negative-threshold", type=float, default=0.30)
    ap.add_argument("--model-path", default="./reranker_lgb.joblib")
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument(
        "--split-mode",
        choices=["time", "random"],
        default="time",
        help="Validation split mode. 'time' uses the last validation fraction by event order.",
    )
    ap.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation fraction.",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading candidates...")
    sasrec = load_i2i_jsonl(args.sasrec_path)
    lfm = load_i2i_jsonl(args.lfm_path)
    hstu_user = load_user_hstu(args.hstu_path)
    track_to_artist, artist_to_tracks = load_tracks(args.tracks_path)

    print(f"SasRec items: {len(sasrec)}")
    print(f"LightFM items: {len(lfm)}")
    print(f"HSTU users: {len(hstu_user)}")
    print(f"Track artists: {len(track_to_artist)}")

    histories: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    X, y = [], []
    events_used = 0
    positive_events = 0
    bad_events = 0
    negative_samples = 0
    skipped_no_history = 0

    for event in iter_log_events(args.log_dir):
        group = get_group(event)
        if args.use_only_control and group not in (None, "C", "control", "Control"):
            continue

        user, consumed_track, baseline_item, t = extract_event(event)
        if user is None or consumed_track is None or baseline_item is None:
            continue

        user_hist = histories[user]

        prev_track = user_hist[-1][0] if user_hist else None
        prev_time = user_hist[-1][1] if user_hist else 0.0

        candidates = set()
        source_info: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        if prev_track is not None:
            add_source_candidates(source_info, candidates, sasrec.get(prev_track, []), "sasrec", args.top_k, anchor=prev_track)
            add_source_candidates(source_info, candidates, lfm.get(prev_track, []), "lfm", args.top_k, anchor=prev_track)
            add_same_artist_candidates(
                source_info,
                candidates,
                prev_track,
                track_to_artist,
                artist_to_tracks,
                args.top_k,
            )
        else:
            skipped_no_history += 1

        # HSTU is user-level and is available even without item history.
        add_source_candidates(source_info, candidates, hstu_user.get(user, []), "hstu", args.top_k)

        # Baseline must always be scored with the same feature builder.
        candidates.add(baseline_item)
        source_info[baseline_item]["source_count"] += 0.0
        baseline_source = dict(source_info.get(baseline_item, {}))

        is_good = float(t) >= args.positive_threshold
        is_bad = float(t) <= args.negative_threshold

        if is_good:
            X.append(build_features(
                baseline_item, baseline_item, prev_track, prev_time, user_hist,
                source_info, baseline_source, track_to_artist, args.history_limit
            ))
            y.append(1)
            positive_events += 1

            negs = [c for c in candidates if int(c) != int(baseline_item)]
            hard = sorted(negs, key=lambda c: source_info[c].get("source_count", 0.0), reverse=True)
            hard = hard[: max(1, args.max_negatives_per_event // 2)]
            rest = [c for c in negs if c not in set(hard)]
            random.shuffle(rest)
            sampled = hard + rest[: max(0, args.max_negatives_per_event - len(hard))]

            for c in sampled:
                X.append(build_features(
                    c, baseline_item, prev_track, prev_time, user_hist,
                    source_info, baseline_source, track_to_artist, args.history_limit
                ))
                y.append(0)
                negative_samples += 1

        elif is_bad:
            # Bad event: exposed baseline is a real negative.
            X.append(build_features(
                baseline_item, baseline_item, prev_track, prev_time, user_hist,
                source_info, baseline_source, track_to_artist, args.history_limit
            ))
            y.append(0)
            bad_events += 1

            negs = [c for c in candidates if int(c) != int(baseline_item)]
            random.shuffle(negs)
            for c in negs[: max(1, args.max_negatives_per_event // 4)]:
                X.append(build_features(
                    c, baseline_item, prev_track, prev_time, user_hist,
                    source_info, baseline_source, track_to_artist, args.history_limit
                ))
                y.append(0)
                negative_samples += 1

        # Update history: keep only the last `history_limit` entries to match online behavior.
        user_hist.append((consumed_track, float(t)))
        if len(user_hist) > args.history_limit:
            # Remove oldest entries (keep latest args.history_limit)
            del user_hist[:len(user_hist) - args.history_limit]

        events_used += 1

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    print(f"Events used: {events_used}")
    print(f"Positive events: {positive_events}")
    print(f"Bad events used as negatives: {bad_events}")
    print(f"Skipped first-history i2i events: {skipped_no_history}")
    print(f"Negative samples: {negative_samples}")
    print(f"Training rows: {len(y)}")
    if len(y) == 0:
        raise RuntimeError("No training rows created.")
    print(f"Positive ratio: {float(y.mean()):.6f}")

    if len(set(y.tolist())) < 2:
        raise RuntimeError("Need both positive and negative samples.")

    # Split data
    if args.split_mode == "time":
        split = int(len(y) * (1.0 - args.val_size))
        split = max(1, min(len(y) - 1, split))

        X_train = X[:split]
        y_train = y[:split]
        X_val = X[split:]
        y_val = y[split:]

        print(f"Validation split: time-ordered, train={len(y_train)}, val={len(y_val)}")
        print(f"Train positive ratio: {float(y_train.mean()):.6f}")
        print(f"Val positive ratio: {float(y_val.mean()):.6f}")

        if len(set(y_train.tolist())) < 2 or len(set(y_val.tolist())) < 2:
            print("[WARN] Time split produced a single-class train or validation set.")
            print("[WARN] Falling back to random stratified split.")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=args.val_size, random_state=args.seed, stratify=y
            )
            print(f"Validation split: random-stratified fallback, train={len(y_train)}, val={len(y_val)}")
            print(f"Train positive ratio: {float(y_train.mean()):.6f}")
            print(f"Val positive ratio: {float(y_val.mean()):.6f}")
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=args.val_size, random_state=args.seed, stratify=y
        )
        print(f"Validation split: random-stratified, train={len(y_train)}, val={len(y_val)}")
        print(f"Train positive ratio: {float(y_train.mean()):.6f}")
        print(f"Val positive ratio: {float(y_val.mean()):.6f}")

    # LightGBM 
    if HAS_LGBM:
        clf = LGBMClassifier(
            n_estimators=80,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
            min_child_samples=20,
        )
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            random_state=args.seed,
        )

    clf.fit(X_train, y_train)

    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X_val)[:, 1]
    else:
        p = clf.predict(X_val)

    auc = roc_auc_score(y_val, p)
    ll = log_loss(y_val, np.clip(p, 1e-6, 1 - 1e-6))

    print(f"Validation AUC: {auc:.6f}")
    print(f"Validation logloss: {ll:.6f}")
    print(
        "Prediction stats: "
        f"min={p.min():.6f}, "
        f"p50={np.median(p):.6f}, "
        f"p90={np.quantile(p, 0.9):.6f}, "
        f"max={p.max():.6f}"
    )

    def describe_scores(name, scores):
        scores = np.asarray(scores)
        if scores.size == 0:
            print(f"{name}: empty")
            return
        print(
            f"{name}: "
            f"min={np.min(scores):.6f}, "
            f"p10={np.quantile(scores, 0.10):.6f}, "
            f"p50={np.quantile(scores, 0.50):.6f}, "
            f"p90={np.quantile(scores, 0.90):.6f}, "
            f"max={np.max(scores):.6f}"
        )

    describe_scores("Val positive scores", p[y_val == 1])
    describe_scores("Val negative scores", p[y_val == 0])

    print("Feature distribution diagnostics on train split:")
    feature_index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    diagnostic_cols = [
        "sasrec_hit",
        "lfm_hit",
        "hstu_hit",
        "same_artist_hit",
        "source_count",
        "sasrec_rank_inv",
        "lfm_rank_inv",
        "hstu_rank_inv",
        "same_artist_rank_inv",
        "diff_source_count_vs_baseline",
        "diff_total_rank_inv_vs_baseline",
        "diff_hstu_rank_inv_vs_baseline",
    ]
    for col in diagnostic_cols:
        if col not in feature_index:
            continue
        idx = feature_index[col]
        pos_vals = X_train[y_train == 1, idx]
        neg_vals = X_train[y_train == 0, idx]
        pos_mean = float(pos_vals.mean()) if pos_vals.size else float("nan")
        neg_mean = float(neg_vals.mean()) if neg_vals.size else float("nan")
        pos_p90 = float(np.quantile(pos_vals, 0.90)) if pos_vals.size else float("nan")
        neg_p90 = float(np.quantile(neg_vals, 0.90)) if neg_vals.size else float("nan")
        print(
            f"{col}: "
            f"pos_mean={pos_mean:.6f}, neg_mean={neg_mean:.6f}, "
            f"pos_p90={pos_p90:.6f}, neg_p90={neg_p90:.6f}"
        )

    bundle = {
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "hstu_mode": "user_to_tracks",
        "top_k": args.top_k,
        "history_limit": args.history_limit,   # store to allow online service validation
        "positive_threshold": args.positive_threshold,
        "negative_threshold": args.negative_threshold,
    }

    joblib.dump(bundle, args.model_path)
    print(f"Saved model bundle to {args.model_path}")


if __name__ == "__main__":
    main()