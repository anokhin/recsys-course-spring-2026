from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path


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
        return -1


def main():
    from botify.recommenders.content_rerank import ContentRerankRecommender

    data_dir = ROOT / "data"
    embeddings_path = data_dir / "content_embeddings.npy"
    meta_path = data_dir / "content_embeddings_meta.json"
    if not embeddings_path.exists():
        raise SystemExit(f"missing {embeddings_path}; run scripts/build_content_embeddings.py first")

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

    out = rec.recommend_next(user=42, prev_track=10, prev_track_time=0.0)
    assert fallback.calls == 1
    print("[ok] empty history -> fallback")

    session = [(100, 0.9), (101, 0.7), (102, 0.05), (103, 0.85)]
    for tr, tm in reversed(session):
        listen_redis.lpush("user:42:listens", json.dumps({"track": tr, "time": tm}))

    pick = rec.recommend_next(user=42, prev_track=103, prev_track_time=0.85)
    print(f"[ok] pick = {pick}")
    assert pick != -1
    assert pick not in {t for t, _ in session}

    print("ALL OK")


if __name__ == "__main__":
    main()
