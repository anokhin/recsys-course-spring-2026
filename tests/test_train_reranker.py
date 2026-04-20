import pandas as pd

from script.train_reranker import (
    sessionize_log,
    build_candidate_features,
    extract_training_rows,
)


def _mk_row(ts, user, track, time, msg, rec=None, exp="RERANKER", treatment="T1"):
    return {
        "timestamp": ts,
        "user": user,
        "track": track,
        "time": time,
        "message": msg,
        "recommendation": rec,
        "experiments": {exp: treatment},
    }


def test_sessionize_splits_on_last():
    df = pd.DataFrame([
        _mk_row(1, 7, 10, 0.5, "next", rec=20),
        _mk_row(2, 7, 20, 0.9, "next", rec=30),
        _mk_row(3, 7, 30, 0.1, "last"),
        _mk_row(4, 7, 40, 0.6, "next", rec=50),
        _mk_row(5, 7, 50, 0.8, "last"),
    ])
    sessions = sessionize_log(df)
    assert len(sessions) == 2
    assert sessions[0]["user"] == 7
    assert sessions[0]["events"][0]["track"] == 10
    assert sessions[0]["events"][-1]["track"] == 30


def test_build_features_matches_expected_columns():
    session = {
        "user": 7,
        "events": [
            {"track": 10, "time": 0.5, "recommendation": 20, "message": "next"},
            {"track": 20, "time": 0.9, "recommendation": 30, "message": "next"},
            {"track": 30, "time": 0.3, "recommendation": None, "message": "last"},
        ],
    }
    track_meta = {10: {"artist": "a"}, 20: {"artist": "a"}, 30: {"artist": "b"}}
    rows = extract_training_rows(session, track_meta)
    # Two 'next' events with recommendation -> two training rows
    assert len(rows) == 2
    keys = set(rows[0])
    assert {"user", "candidate", "history_len", "mean_prev_time",
            "prev_artist_match", "candidate_rank", "label"} <= keys


def test_build_candidate_features_shape():
    feats = build_candidate_features(
        history=[(10, 0.5), (20, 0.9)],
        candidate=30,
        candidate_rank=2,
        track_meta={10: {"artist": "a"}, 20: {"artist": "a"}, 30: {"artist": "b"}},
    )
    assert feats["history_len"] == 2
    assert abs(feats["mean_prev_time"] - 0.7) < 1e-6
    assert feats["prev_artist_match"] == 0
    assert feats["candidate_rank"] == 2
