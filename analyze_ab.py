import argparse
import glob
import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as ss

LABEL_CONTROL = "C"
METRICS = [
    "time",
    "sessions",
    "mean_request_latency",
    "mean_tracks_per_session",
    "mean_time_per_session",
]


def read_logs(data_dir: Path) -> pd.DataFrame:
    paths = (
            glob.glob(str(data_dir / "*/data.json")) or
            glob.glob(str(data_dir / "**/data.json"), recursive=True)
    )
    if not paths:
        raise FileNotFoundError(f"Не найдено data.json в {data_dir}")
    return pd.concat([pd.read_json(p, lines=True) for p in sorted(paths)])


def detect_experiment(df: pd.DataFrame) -> str:
    keys = set(
        k for e in df["experiments"]
        if isinstance(e, dict)
        for k in e
    )
    exp = sorted(keys)[0]
    print(f"  Эксперимент: '{exp}'")
    return exp


Session = namedtuple("Session", ["timestamp", "tracks", "time", "latency"])


def sessionize(user_data):
    sessions, session = [], None
    for _, row in user_data.sort_values("timestamp").iterrows():
        if session is None:
            session = Session(row["timestamp"], 0, 0, 0)
        session = session._replace(
            tracks=session.tracks + 1,
            time=session.time + row["time"],
            latency=session.latency + row["latency"] * 1000,
        )
        if row["message"] == "last":
            sessions.append(session._asdict())
            session = None
    return sessions


def build_user_level_data(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    df = df.copy()
    df["treatment"] = df["experiments"].map(
        lambda e: e.get(experiment) if isinstance(e, dict) else None
    )
    df = df.dropna(subset=["treatment"])

    sessions = (
        df.groupby(["user", "treatment"])
        .apply(sessionize)
        .explode()
        .apply(pd.Series)
    )
    user_data = (
        sessions.reset_index()
        .groupby(["user", "treatment"])
        .agg({"timestamp": "count", "tracks": "sum",
              "time": "sum", "latency": "sum"})
    )
    user_data["sessions"] = user_data["timestamp"]
    user_data["mean_request_latency"] = user_data["latency"] / user_data["tracks"]
    user_data["mean_tracks_per_session"] = user_data["tracks"] / user_data["sessions"]
    user_data["mean_time_per_session"] = user_data["time"] / user_data["sessions"]
    return user_data[METRICS].copy().reset_index()


def _dof(n0, n1, s2_0, s2_1):
    if n0 <= 1 or n1 <= 1:
        return np.inf
    num = (s2_0 / n0 + s2_1 / n1) ** 2
    den = s2_0 ** 2 / n0 ** 2 / (n0 - 1) + s2_1 ** 2 / n1 ** 2 / (n1 - 1)
    if den <= 0 or not np.isfinite(num) or not np.isfinite(den):
        return np.inf
    return num / den


def _ci(n0, n1, s2_0, s2_1, alpha=0.05):
    if not np.isfinite(s2_0):
        s2_0 = 0.0
    if not np.isfinite(s2_1):
        s2_1 = 0.0

    scale = s2_0 / n0 + s2_1 / n1
    if scale <= 0 or not np.isfinite(scale):
        return 0.0

    dof = _dof(n0, n1, s2_0, s2_1)
    if dof == np.inf:
        critical = ss.norm.ppf(1 - alpha / 2)
    else:
        critical = ss.t.ppf(1 - alpha / 2, dof)

    if not np.isfinite(critical):
        return 0.0
    return float(critical * np.sqrt(scale))


def compute_effects(user_metrics: pd.DataFrame) -> list:
    agg = user_metrics.groupby("treatment")[METRICS].agg(["count", "mean", "var"])
    ctrl = agg.loc[LABEL_CONTROL]
    effects = []
    for treatment, row in agg.iterrows():
        if treatment == LABEL_CONTROL:
            continue
        for metric in METRICS:
            c_mean = ctrl[metric]["mean"]
            t_mean = row[metric]["mean"]
            effect = t_mean - c_mean
            conf_int = _ci(
                ctrl[metric]["count"], row[metric]["count"],
                ctrl[metric]["var"], row[metric]["var"],
            )
            effect_pct = float(effect / c_mean * 100) if c_mean else 0.0
            lower_pct = float((effect - conf_int) / c_mean * 100) if c_mean else 0.0
            upper_pct = float((effect + conf_int) / c_mean * 100) if c_mean else 0.0
            effects.append({
                "treatment": treatment,
                "metric": metric,
                "control_mean": round(float(c_mean), 4),
                "treatment_mean": round(float(t_mean), 4),
                "effect_pct": round(effect_pct, 2) if np.isfinite(effect_pct) else 0.0,
                "lower_pct": round(lower_pct, 2) if np.isfinite(lower_pct) else 0.0,
                "upper_pct": round(upper_pct, 2) if np.isfinite(upper_pct) else 0.0,
                "significant": bool((effect + conf_int) * (effect - conf_int) > 0),
            })
    return effects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = read_logs(Path(args.data))
    experiment = detect_experiment(df)
    user_metrics = build_user_level_data(df, experiment)
    effects = compute_effects(user_metrics)

    print(
        pd.DataFrame(effects)[[
            "treatment", "metric", "effect_pct",
            "upper_pct", "lower_pct",
            "control_mean", "treatment_mean", "significant"
        ]]
        .sort_values(["metric", "treatment"], ascending=False)
        .to_string(index=False)
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"all_effects": effects}, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"\n💾 {out}")


if __name__ == "__main__":
    main()
