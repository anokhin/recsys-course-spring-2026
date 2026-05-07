import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


GOOD_TIME = 0.70
SKIP_TIME = 0.20
MISSING_RANK = 0


def read_json_or_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return rows


def to_int_or_none(x):
    try:
        return int(x)
    except Exception:
        return None


def extract_recommendation_mapping(obj):
    mapping = {}

    def add(key, recs):
        key = to_int_or_none(key)
        if key is None or recs is None:
            return

        if isinstance(recs, dict):
            for field in ["recommendations", "tracks", "items", "recs"]:
                if field in recs:
                    recs = recs[field]
                    break

        if not isinstance(recs, list):
            return

        clean = []
        for r in recs:
            if isinstance(r, dict):
                for field in ["track", "item", "id"]:
                    if field in r:
                        r = r[field]
                        break

            r = to_int_or_none(r)
            if r is not None:
                clean.append(r)

        if clean:
            mapping[key] = clean

    if isinstance(obj, dict):
        for key, value in obj.items():
            add(key, value)

    elif isinstance(obj, list):
        for row in obj:
            if not isinstance(row, dict):
                continue

            key = None
            for key_field in [
                "user",
                "user_id",
                "track",
                "track_id",
                "item",
                "item_id",
                "id",
            ]:
                if key_field in row:
                    key = row[key_field]
                    break

            recs = None
            for rec_field in [
                "recommendations",
                "tracks",
                "items",
                "recs",
                "candidates",
            ]:
                if rec_field in row:
                    recs = row[rec_field]
                    break

            add(key, recs)

    return mapping

def find_recommendation_file(data_dir: Path, keywords):
    candidates = []

    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue

        name = path.name.lower()
        if all(k.lower() in name for k in keywords):
            candidates.append(path)

    if not candidates:
        return None

    return sorted(candidates, key=lambda p: len(str(p)))[0]


def load_recommendation_mapping(data_dir: Path, explicit_path: str, keywords, name: str):
    if explicit_path:
        path = Path(explicit_path)
    else:
        path = find_recommendation_file(data_dir, keywords)

    if path is None or not path.exists():
        print(f"[WARN] Could not find {name} recommendations. keywords={keywords}")
        return {}

    print(f"Loading {name} recommendations from {path}")
    obj = read_json_or_jsonl(path)
    mapping = extract_recommendation_mapping(obj)
    print(f"Loaded {len(mapping)} keys for {name}")
    return mapping


def read_logs(log_dir: Path) -> pd.DataFrame:
    rows = []

    patterns = ["data.json*", "*.jsonl"]
    log_files = []

    for pattern in patterns:
        log_files.extend(list(log_dir.rglob(pattern)))

    log_files = sorted(set(log_files))

    if not log_files:
        raise RuntimeError(f"No log files found under {log_dir}")

    for path in log_files:
        if not path.is_file():
            continue

        print(f"Reading {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if obj.get("message") not in {"next", "last"}:
                    continue

                exp = obj.get("experiments", {})

                rows.append(
                    {
                        "message": obj.get("message"),
                        "timestamp": obj.get("timestamp"),
                        "user": obj.get("user"),
                        "track": obj.get("track"),
                        "time": obj.get("time", 0.0),
                        "recommendation": obj.get("recommendation"),
                        "group": exp.get("HSTU"),
                    }
                )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No valid logs found.")

    df = df.dropna(subset=["timestamp", "user", "track"])
    df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)

    return df


def add_candidate(candidate_info, track, source, rank):
    track = to_int_or_none(track)
    if track is None:
        return

    rank = int(rank) + 1

    if track not in candidate_info:
        candidate_info[track] = {
            "sasrec_rank": MISSING_RANK,
            "hstu_rank": MISSING_RANK,
            "lfm_rank": MISSING_RANK,
        }

    field = f"{source}_rank"
    old = candidate_info[track].get(field, MISSING_RANK)

    if old == MISSING_RANK:
        candidate_info[track][field] = rank
    else:
        candidate_info[track][field] = min(old, rank)

def build_candidate_info(user, prev_track, sasrec_map, hstu_map, lfm_map, top_k):
    candidate_info = {}

    sasrec_recs = sasrec_map.get(int(prev_track), [])
    for rank, track in enumerate(sasrec_recs[:top_k]):
        add_candidate(candidate_info, track, "sasrec", rank)

    hstu_recs = hstu_map.get(int(user), [])
    for rank, track in enumerate(hstu_recs[:top_k]):
        add_candidate(candidate_info, track, "hstu", rank)

    lfm_recs = lfm_map.get(int(prev_track), [])
    for rank, track in enumerate(lfm_recs[:top_k]):
        add_candidate(candidate_info, track, "lfm", rank)

    return candidate_info

# def build_candidate_info(user, prev_track, sasrec_map, hstu_map, lfm_map, top_k):
#     candidate_info = {}

#     sasrec_recs = sasrec_map.get(int(prev_track), [])
#     for rank, track in enumerate(sasrec_recs[:top_k]):
#         add_candidate(candidate_info, track, "sasrec", rank)

#     return candidate_info

def make_session_features(g, i):
    cur = g.iloc[i]
    recent = g.iloc[max(0, i - 5): i + 1]
    recent_times = recent["time"].astype(float).values

    hist_len = len(recent)
    recent_avg_time = float(np.mean(recent_times)) if hist_len else 0.0
    recent_good_frac = float(np.mean(recent_times >= GOOD_TIME)) if hist_len else 0.0
    recent_skip_frac = float(np.mean(recent_times <= SKIP_TIME)) if hist_len else 0.0

    return {
        "prev_time": float(cur["time"]),
        "hist_len": hist_len,
        "recent_avg_time": recent_avg_time,
        "recent_last_time": float(cur["time"]),
        "recent_good_frac": recent_good_frac,
        "recent_skip_frac": recent_skip_frac,
    }


def build_training_data(
    df: pd.DataFrame,
    sasrec_map,
    hstu_map,
    lfm_map,
    use_only_control: bool = True,
    top_k: int = 20,
    max_negatives_per_event: int = 30,
) -> pd.DataFrame:
    if use_only_control:
        df = df[df["group"] == "C"].copy()

    samples = []

    for user, g in df.groupby("user", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)

        for i in range(len(g) - 1):
            cur = g.iloc[i]
            nxt = g.iloc[i + 1]

            if cur["message"] != "next":
                continue

            user_id = int(user)
            prev_track = int(cur["track"])
            true_next = int(nxt["track"])
            true_next_time = float(nxt["time"])
            is_good_next = true_next_time >= GOOD_TIME

            candidate_info = build_candidate_info(
                user=user_id,
                prev_track=prev_track,
                sasrec_map=sasrec_map,
                hstu_map=hstu_map,
                lfm_map=lfm_map,
                top_k=top_k,
            )

            if not candidate_info:
                continue

            session_features = make_session_features(g, i)

            positives = []
            negatives = []

            for candidate, info in candidate_info.items():
                label = int(candidate == true_next and is_good_next)

                source_count = int(info.get("sasrec_rank", 0) > 0)
                source_count += int(info.get("hstu_rank", 0) > 0)
                source_count += int(info.get("lfm_rank", 0) > 0)

                row = {
                    **session_features,
                    "sasrec_rank": int(info.get("sasrec_rank", MISSING_RANK)),
                    "hstu_rank": int(info.get("hstu_rank", MISSING_RANK)),
                    "lfm_rank": int(info.get("lfm_rank", MISSING_RANK)),
                    "source_count": source_count,
                    "label": label,
                    "user_group": user_id,
                }

                if label == 1:
                    positives.append(row)
                else:
                    negatives.append(row)

            if max_negatives_per_event > 0 and len(negatives) > max_negatives_per_event:
                negatives = random.sample(negatives, max_negatives_per_event)

            samples.extend(positives)
            samples.extend(negatives)

    data = pd.DataFrame(samples)

    if data.empty:
        raise RuntimeError("No training samples built.")

    return data

def train_logistic(data: pd.DataFrame, model_path: Path):
    if model_path.suffix.lower() != ".json":
        print(
            f"[WARN] Model path {model_path} does not end with .json. "
            "The model will still be saved as plain JSON, not joblib."
        )

    features = [
        "prev_time",
        "hist_len",
        "recent_avg_time",
        "recent_last_time",
        "recent_good_frac",
        "recent_skip_frac",
        "sasrec_rank",
        "hstu_rank",
        "lfm_rank",
        "source_count",
    ]

    X = data[features]
    y = data["label"]
    groups = data["user_group"].values

    print("Training samples:", len(data))
    print("Positive rate:", float(y.mean()))

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, valid_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]
    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    pred = model.predict_proba(X_valid)[:, 1]

    if y_valid.nunique() > 1:
        auc = roc_auc_score(y_valid, pred)
        print("Validation AUC:", auc)

    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]

    model_data = {
        "features": features,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "params": {
            "good_time": GOOD_TIME,
            "skip_time": SKIP_TIME,
            "description": "Session-aware Logistic Residual Reranker",
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False, indent=2)

    print(f"Saved JSON model to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./logistic_reranker_model.json",
    )
    parser.add_argument("--use-only-control", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-negatives-per-event", type=int, default=30)
    parser.add_argument("--sasrec-path", type=str, default="")
    parser.add_argument("--hstu-path", type=str, default="")
    parser.add_argument("--lfm-path", type=str, default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    sasrec_map = load_recommendation_mapping(
        data_dir=data_dir,
        explicit_path=args.sasrec_path,
        keywords=["sasrec"],
        name="SasRec",
    )

    hstu_map = load_recommendation_mapping(
        data_dir=data_dir,
        explicit_path=args.hstu_path,
        keywords=["hstu"],
        name="HSTU",
    )

    lfm_map = load_recommendation_mapping(
        data_dir=data_dir,
        explicit_path=args.lfm_path,
        keywords=["lightfm"],
        name="LightFM",
    )

    logs = read_logs(Path(args.log_dir))
    print("Raw logs:", len(logs))
    print(logs["group"].value_counts(dropna=False))

    data = build_training_data(
        logs,
        sasrec_map=sasrec_map,
        hstu_map=hstu_map,
        lfm_map=lfm_map,
        use_only_control=args.use_only_control,
        top_k=args.top_k,
        max_negatives_per_event=args.max_negatives_per_event,
    )

    train_logistic(data, Path(args.model_path))


if __name__ == "__main__":
    main()