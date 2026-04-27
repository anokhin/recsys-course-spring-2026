"""Smoke test for ContentRerankRecommender.

Run with:
    cd botify && PYTHONPATH=. python tests/test_content_rerank.py

Uses an in-memory fake Redis so it doesn't need a real server.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path


# Make `botify.*` importable when running this file directly.
HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))


class FakeRedis:
    def __init__(self):
        self._kv: dict[bytes | str, object] = {}
        self._lists: dict[str, list] = defaultdict(list)

    def set(self, key, value):
        if isinstance(key, int):
            key = str(key)
        self._kv[key] = value

    def get(self, key):
        if isinstance(key, int):
            key = str(key)
        return self._kv.get(key)

    def lpush(self, key, value):
        self._lists[key].insert(0, value)

    def lrange(self, key, start, end):
        items = self._lists.get(key, [])
        if end < 0:
            end = len(items) + end
        return items[start : end + 1]

    def ltrim(self, key, start, end):
        items = self._lists.get(key, [])
        self._lists[key] = items[start : end + 1]


class FallbackRecommender:
    def __init__(self):
        self.calls = 0

    def recommend_next(self, user, prev_track, prev_track_time):
        self.calls += 1
        return -1  # sentinel


def main():
    from botify.recommenders.content_rerank import ContentRerankRecommender

    data_dir = ROOT / "data"
    embeddings_path = data_dir / "content_embeddings.npy"
    meta_path = data_dir / "content_embeddings_meta.json"
    if not embeddings_path.exists():
        raise SystemExit(f"missing {embeddings_path}; run `python scripts/build_content_embeddings.py` first")

    # SasRec-I2I: load a few entries
    sasrec_redis = FakeRedis()
    with open(data_dir / "sasrec_i2i.jsonl") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            sasrec_redis.set(int(r["item_id"]), pickle.dumps(r["recommendations"]))
            if i > 5000:
                break

    listen_redis = FakeRedis()
    fallback = FallbackRecommender()

    rec = ContentRerankRecommender(
        listen_history_redis=listen_redis,
        i2i_redis=sasrec_redis,
        embeddings_path=str(embeddings_path),
        meta_path=str(meta_path),
        fallback_recommender=fallback,
    )

    # Case 1: empty history -> falls back
    out = rec.recommend_next(user=42, prev_track=10, prev_track_time=0.0)
    assert fallback.calls == 1, "empty history should fall back"
    print("[ok] empty history -> fallback")

    # Case 2: synthesise a session — user listened to some tracks
    session = [(100, 0.9), (101, 0.7), (102, 0.05), (103, 0.85)]
    for tr, tm in reversed(session):
        listen_redis.lpush("user:42:listens", json.dumps({"track": tr, "time": tm}))

    pick = rec.recommend_next(user=42, prev_track=103, prev_track_time=0.85)
    print(f"[ok] history-based pick = {pick}")
    assert pick != -1, "should not have fallen back when sasrec has neighbours"
    assert pick not in {t for t, _ in session}, "must not recommend already-heard track"

    print("\nALL OK")


if __name__ == "__main__":
    main()
