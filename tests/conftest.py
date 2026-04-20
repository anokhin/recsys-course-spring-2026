import json
import pickle
from collections import defaultdict

import pytest


class FakeRedis:
    """In-memory stand-in for a redis connection used by botify recommenders."""

    def __init__(self):
        self._kv = {}
        self._lists = defaultdict(list)

    def get(self, key):
        if isinstance(key, int):
            key = str(key)
        return self._kv.get(key)

    def set(self, key, value):
        if isinstance(key, int):
            key = str(key)
        self._kv[key] = value

    def lpush(self, key, value):
        self._lists[key].insert(0, value)

    def ltrim(self, key, start, stop):
        self._lists[key] = self._lists[key][start:stop + 1]

    def lrange(self, key, start, stop):
        lst = self._lists[key]
        if stop == -1:
            return lst[start:]
        return lst[start:stop + 1]


class FakeCatalog:
    def from_bytes(self, blob):
        return pickle.loads(blob)

    def to_bytes(self, obj):
        return pickle.dumps(obj)


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def fake_catalog():
    return FakeCatalog()


@pytest.fixture
def push_history(fake_redis):
    def _push(user, events):
        key = f"user:{user}:listens"
        for track, t in events:
            fake_redis.lpush(key, json.dumps({"track": track, "time": t}))
    return _push


@pytest.fixture
def set_sasrec_recs(fake_redis):
    """Pickle-encode track recommendations into FakeRedis the way Catalog does."""
    def _set(track_id, recs):
        fake_redis.set(track_id, pickle.dumps(recs))
    return _set
