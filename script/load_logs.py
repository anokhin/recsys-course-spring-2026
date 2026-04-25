"""Parse simulator data.json logs into a tidy DataFrame, filtered to one experiment arm."""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import List

import pandas as pd


def load_control_arm(
    data_dir: str | Path,
    experiment_name: str = "HSTU",
    control_label: str = "C",
    aggregate: bool = False,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    paths: List[str] = sorted(
        glob.glob(str(data_dir / "**" / "data.json"), recursive=True)
    )
    if not paths:
        raise FileNotFoundError(f"No data.json under {data_dir}")

    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                row = json.loads(line)
                exp = row.get("experiments") or {}
                if exp.get(experiment_name) != control_label:
                    continue
                user = row.get("user")
                track = row.get("track")
                listen_time = row.get("time")
                if user is None or track is None or listen_time is None:
                    continue
                rows.append((int(user), int(track), float(listen_time)))

    df = pd.DataFrame(rows, columns=["user", "track", "listen_time"])
    if aggregate and not df.empty:
        df = df.groupby(["user", "track"], as_index=False)["listen_time"].sum()
    return df
