import json

import numpy as np
import pandas as pd


def test_train_pipeline_smoke(tmp_path):
    """Full train_and_recommend on a tiny synthetic dataset writes a valid JSONL."""
    from script.train_recsys import train_and_recommend

    rng = np.random.default_rng(0)
    rows = []
    for user in range(4):
        for track in rng.choice(8, size=4, replace=False):
            rows.append((user, int(track), float(rng.uniform(0.1, 1.0))))
    logs = pd.DataFrame(rows, columns=["user", "track", "listen_time"])

    embeddings = rng.standard_normal(size=(8, 16)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    out_path = tmp_path / "recs.jsonl"
    train_and_recommend(
        logs=logs,
        track_embeddings=embeddings,
        all_user_ids=[0, 1, 2, 3, 4],
        out_path=out_path,
        top_n=3,
        ials_factors=4,
        ials_iterations=3,
        ials_top_k=6,
        catboost_iterations=20,
        seed=42,
    )

    with open(out_path) as f:
        recs = [json.loads(line) for line in f]
    assert {r["user"] for r in recs} == {0, 1, 2, 3, 4}
    for r in recs:
        assert isinstance(r["user"], int)
        assert isinstance(r["tracks"], list)
        assert len(r["tracks"]) == 3
        assert all(0 <= t < 8 for t in r["tracks"])
