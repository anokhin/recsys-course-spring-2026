import json
from pathlib import Path

from script.load_logs import load_control_arm


def _write_log(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_control_arm_filters_by_experiment(tmp_path):
    rows = [
        {"message": "next", "user": 1, "track": 10, "time": 0.8,
         "experiments": {"HSTU": "C"}},
        {"message": "next", "user": 2, "track": 11, "time": 0.3,
         "experiments": {"HSTU": "T1"}},
        {"message": "last", "user": 1, "track": 12, "time": 0.5,
         "experiments": {"HSTU": "C"}},
    ]
    _write_log(tmp_path / "rec1" / "data.json", rows)

    df = load_control_arm(tmp_path, experiment_name="HSTU", control_label="C")

    assert set(df.columns) >= {"user", "track", "listen_time"}
    assert len(df) == 2
    assert sorted(df["track"].tolist()) == [10, 12]
    assert df.loc[df["track"] == 10, "listen_time"].iloc[0] == 0.8


def test_load_control_arm_aggregates_per_user_track(tmp_path):
    rows = [
        {"message": "next", "user": 1, "track": 5, "time": 0.4,
         "experiments": {"HSTU": "C"}},
        {"message": "next", "user": 1, "track": 5, "time": 0.6,
         "experiments": {"HSTU": "C"}},
    ]
    _write_log(tmp_path / "rec1" / "data.json", rows)

    df = load_control_arm(
        tmp_path, experiment_name="HSTU", control_label="C", aggregate=True
    )
    assert len(df) == 1
    assert df.iloc[0]["listen_time"] == 1.0
