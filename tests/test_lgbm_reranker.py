import numpy as np
import pytest

from botify.recommenders.lgbm_reranker import LGBMReranker


class DummyBooster:
    """Deterministic stand-in: score = history_len - 0.01 * candidate_rank."""
    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] - X[:, 4] * 0.01


class _StaticFallback:
    def __init__(self, val):
        self.val = val

    def recommend_next(self, u, p, t):
        return self.val


FEATURE_ORDER = [
    "history_len", "mean_prev_time", "last_time",
    "prev_artist_match", "candidate_rank",
    "i2v_cos_mean", "i2v_cos_last", "candidate_seen_before",
]


@pytest.fixture
def reranker(fake_redis):
    return LGBMReranker(
        listen_history_redis=fake_redis,
        sasrec_redis=fake_redis,
        fallback=_StaticFallback(999),
        booster=DummyBooster(),
        feature_order=FEATURE_ORDER,
        track_meta={10: {"artist": "a"}, 100: {"artist": "x"},
                    101: {"artist": "a"}, 102: {"artist": "y"}},
        item2vec={},
        top_k=3,
    )


def test_picks_argmax_candidate(reranker, push_history, set_sasrec_recs):
    push_history(1, [(10, 0.9)])
    set_sasrec_recs(10, [100, 101, 102])
    chosen = reranker.recommend_next(1, 10, 0.9)
    # all candidates have same history_len; rank 0 wins via -0.01*rank
    assert chosen == 100


def test_skips_seen(reranker, push_history, set_sasrec_recs):
    push_history(1, [(10, 0.9), (100, 0.1)])
    set_sasrec_recs(10, [100, 101, 102])
    assert reranker.recommend_next(1, 10, 0.9) != 100


def test_fallback_when_no_history(fake_redis):
    rer = LGBMReranker(
        listen_history_redis=fake_redis,
        sasrec_redis=fake_redis,
        fallback=_StaticFallback(999),
        booster=DummyBooster(),
        feature_order=FEATURE_ORDER,
        track_meta={},
        item2vec={},
        top_k=3,
    )
    assert rer.recommend_next(1, 10, 0.9) == 999


def test_deterministic(reranker, push_history, set_sasrec_recs):
    push_history(1, [(10, 0.9)])
    set_sasrec_recs(10, [100, 101, 102])
    a = reranker.recommend_next(1, 10, 0.9)
    b = reranker.recommend_next(1, 10, 0.9)
    assert a == b


def test_factory_returns_fallback_when_missing(tmp_path, monkeypatch, fake_redis):
    from botify.recommenders import reranker_factory

    monkeypatch.setenv("RERANKER_DIR", str(tmp_path))

    class _F:
        def recommend_next(self, *a):
            return 7

    out = reranker_factory.build_reranker(
        listen_history_redis=fake_redis,
        sasrec_redis=fake_redis,
        fallback=_F(),
        tracks_json_path="/dev/null",
    )
    assert out.recommend_next(1, 2, 3.0) == 7
