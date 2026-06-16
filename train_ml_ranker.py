"""
Offline training script for MLRanker.

The script:
  1. Loads botify event logs collected from a simulator run
  2. Filters Control group events (SasRec-I2I recommendations)
  3. Builds (prev_track, recommendation) pairs with listen-time labels
  4. Trains a GradientBoostingClassifier on 5 features
  5. Saves the model bundle to the output path
"""

import argparse
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

FEATURE_NAMES = ["prev_time", "same_artist", "sasrec_rank", "sasrec_rr", "sasrec_candidate_count"]
LABEL_THRESHOLD = 0.5
SASREC_TOP_K = 20


def load_logs(log_dir: str) -> pd.DataFrame:
    patterns = [
        os.path.join(log_dir, "*/data.json*"),
        os.path.join(log_dir, "**/data.json*"),
        os.path.join(log_dir, "data.json*"),
    ]
    paths = []
    for pat in patterns:
        found = glob.glob(pat, recursive=True)
        if found:
            paths = found
            break
    if not paths:
        raise FileNotFoundError(f"No data.json found in {log_dir}")
    print(f"  Loading {len(paths)} log file(s) from {log_dir}")
    frames = []
    for p in sorted(paths):
        frames.append(pd.read_json(p, lines=True))
    df = pd.concat(frames, ignore_index=True)
    print(f"  Total events: {len(df)}")
    return df


def load_sasrec(sasrec_path: str) -> dict:
    sasrec = {}
    with open(sasrec_path) as f:
        for line in f:
            r = json.loads(line)
            sasrec[int(r["item_id"])] = [int(x) for x in r["recommendations"]][:SASREC_TOP_K]
    print(f"  SasRec loaded: {len(sasrec)} items")
    return sasrec


def load_tracks_meta(tracks_path: str) -> dict:
    tracks_meta = {}
    with open(tracks_path) as f:
        for line in f:
            t = json.loads(line)
            tracks_meta[int(t["track"])] = t["artist"]
    print(f"  Track metadata loaded: {len(tracks_meta)} tracks")
    return tracks_meta


def build_features(df: pd.DataFrame, sasrec: dict, tracks_meta: dict) -> pd.DataFrame:
    df = df[df["message"] == "next"].copy()

    def get_control_treatment(experiments):
        if not isinstance(experiments, dict):
            return None
        for key in ("HSTU", "ML_RANKER"):
            val = experiments.get(key)
            if val is not None:
                return val
        return None

    df["treatment"] = df["experiments"].apply(get_control_treatment)
    df = df[df["treatment"] == "C"].copy()
    print(f"  Control events: {len(df)}")

    df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)

    records = []
    skipped = 0
    for _, user_df in df.groupby("user"):
        rows = list(user_df.itertuples(index=False))
        for i in range(len(rows) - 1):
            cur = rows[i]
            nxt = rows[i + 1]

            # Verify recommendation was actually played next
            rec = getattr(cur, "recommendation", None)
            if rec is None or int(nxt.track) != int(rec):
                skipped += 1
                continue

            prev_track = int(cur.track)
            recommendation = int(rec)
            prev_time = float(cur.time)
            label_time = float(nxt.time)

            sasrec_list = sasrec.get(prev_track, [])
            sasrec_count = len(sasrec_list)

            if recommendation in sasrec_list:
                rank = sasrec_list.index(recommendation) + 1
                rr = 1.0 / rank
            else:
                rank, rr = 0, 0.0

            prev_artist = tracks_meta.get(prev_track)
            rec_artist = tracks_meta.get(recommendation)
            same_artist = int(prev_artist is not None and prev_artist == rec_artist)

            records.append({
                "prev_time": prev_time,
                "same_artist": same_artist,
                "sasrec_rank": rank,
                "sasrec_rr": rr,
                "sasrec_candidate_count": sasrec_count,
                "label_bin": int(label_time > LABEL_THRESHOLD),
            })

    print(f"  Training examples: {len(records)}  (skipped: {skipped})")
    return pd.DataFrame(records)


def train(df_feat: pd.DataFrame, n_estimators: int = 100) -> GradientBoostingClassifier:
    X = df_feat[FEATURE_NAMES].values.astype(float)
    y = df_feat["label_bin"].values

    print(f"  Class balance: {y.mean():.3f} positive")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        n_iter_no_change=10,
        validation_fraction=0.1,
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"  Train accuracy: {train_acc:.4f}  Test accuracy: {test_acc:.4f}")
    print(f"  Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"    {name}: {imp:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train MLRanker model")
    parser.add_argument("--logs", required=True, help="Directory with botify log files")
    parser.add_argument(
        "--sasrec", default="botify/data/sasrec_i2i.jsonl", help="SasRec I2I JSONL file"
    )
    parser.add_argument(
        "--tracks", default="botify/data/tracks.json", help="Track catalog JSON file"
    )
    parser.add_argument(
        "--output",
        default="botify/data/ml_ranker_bundle.joblib",
        help="Output path for model bundle",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100, help="Number of boosting trees"
    )
    args = parser.parse_args()

    df = load_logs(args.logs)

    sasrec = load_sasrec(args.sasrec)
    tracks_meta = load_tracks_meta(args.tracks)

    df_feat = build_features(df, sasrec, tracks_meta)

    if len(df_feat) < 100:
        raise ValueError(f"Too few training examples: {len(df_feat)}. Run more episodes first.")

    print(f"Training GradientBoostingClassifier (n_estimators={args.n_estimators})...")
    model = train(df_feat, n_estimators=args.n_estimators)

    bundle = {"model": model, "feature_names": FEATURE_NAMES}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    joblib.dump(bundle, args.output, compress=3)
    size_kb = os.path.getsize(args.output) / 1024
    print(f"\nSaved model bundle to {args.output}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
