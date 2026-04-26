import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class RecommendationArtifacts:
    prev_track_index: pd.Index
    track_index: pd.Index
    prev_track_rankings: pd.DataFrame
    global_ranking: pd.DataFrame


def read_interactions(data_dir: Path) -> pd.DataFrame:
    data_path = Path(data_dir)
    paths = glob.glob(str(data_path / "*/data.json"))
    if not paths:
        paths = glob.glob(str(data_path / "**/data.json"), recursive=True)
    if not paths:
        raise FileNotFoundError("No data.json files found in {0}".format(data_path))

    events = pd.concat(
        [pd.read_json(path, lines=True) for path in sorted(paths)],
        ignore_index=True,
    )
    events = events[
        ["message", "timestamp", "user", "track", "time", "recommendation"]
    ].dropna(subset=["user", "track", "time"])
    events["user"] = events["user"].astype("int32")
    events["track"] = events["track"].astype("int32")
    events["time"] = events["time"].astype("float32")
    events["recommendation"] = events["recommendation"].astype("Float64")
    events = events.sort_values(["user", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )

    events["next_track"] = events.groupby("user", sort=False)["track"].shift(-1)
    events["next_time"] = events.groupby("user", sort=False)["time"].shift(-1)

    actions = events[events["message"] == "next"].copy()
    actions = actions[
        actions["recommendation"].notna() & actions["next_track"].notna()
    ].copy()
    actions["prev_track"] = actions["track"].astype("int32")
    actions["track"] = actions["recommendation"].astype("int32")
    actions["next_track"] = actions["next_track"].astype("int32")
    actions["time"] = actions["next_time"].astype("float32")
    actions = actions[
        (actions["track"] == actions["next_track"]) & (actions["time"] > 0.0)
    ].copy()
    return actions[["user", "prev_track", "track", "time"]].reset_index(drop=True)


def fit_recommendation_artifacts(
    interactions: pd.DataFrame,
) -> RecommendationArtifacts:
    track_stats = (
        interactions.groupby("track", sort=False)["time"]
        .agg(track_mean="mean", track_total="sum", track_count="size")
        .reset_index()
    )
    track_stats["score"] = track_stats["track_total"]

    prev_track_rankings = (
        interactions.groupby(["prev_track", "track"], sort=False)["time"]
        .agg(sum_time="sum", count="size")
        .reset_index()
    )
    prev_track_rankings["score"] = prev_track_rankings["sum_time"]
    prev_track_rankings = prev_track_rankings.sort_values(
        ["prev_track", "score", "count", "track"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    global_ranking = track_stats.sort_values(
        ["score", "track_count", "track"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return RecommendationArtifacts(
        prev_track_index=pd.Index(interactions["prev_track"].drop_duplicates()),
        track_index=pd.Index(track_stats["track"].drop_duplicates()),
        prev_track_rankings=prev_track_rankings,
        global_ranking=global_ranking,
    )


def build_recommendation_rows(
    artifacts: RecommendationArtifacts,
    top_k: int,
) -> List[Dict[str, object]]:
    rows = [_build_payload_row("global", artifacts.global_ranking, top_k)]

    for prev_track, group in artifacts.prev_track_rankings.groupby("prev_track", sort=False):
        rows.append(_build_payload_row("track:{0}".format(int(prev_track)), group, top_k))

    return rows


def write_recommendation_rows(rows: List[Dict[str, object]], output_path: Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row) + "\n")


def _build_payload_row(key: str, frame: pd.DataFrame, top_k: int) -> Dict[str, object]:
    top = frame.head(top_k)
    return {
        "key": key,
        "payload": {
            "tracks": [int(track) for track in top["track"].tolist()],
            "scores": [float(score) for score in top["score"].tolist()],
        },
    }
