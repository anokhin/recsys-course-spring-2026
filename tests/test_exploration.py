import random
from collections import Counter

from botify.recommenders.exploration import ExplorationRecommender


def test_returns_top1_when_epsilon_zero(fake_redis, push_history, set_sasrec_recs):
    push_history(user=1, events=[(10, 30.0)])
    set_sasrec_recs(10, [100, 101, 102])

    rec = ExplorationRecommender(
        listen_history_redis=fake_redis,
        i2i_redis=fake_redis,
        fallback=None,
        epsilon=0.0,
        top_k=3,
        seed=7,
    )
    result = rec.recommend_next(user=1, prev_track=10, prev_track_time=30.0)
    assert result == 100


def test_explores_with_epsilon_one(fake_redis, push_history, set_sasrec_recs):
    push_history(user=1, events=[(10, 30.0)])
    set_sasrec_recs(10, [100, 101, 102, 103])

    rec = ExplorationRecommender(
        listen_history_redis=fake_redis,
        i2i_redis=fake_redis,
        fallback=None,
        epsilon=1.0,
        top_k=4,
        seed=0,
    )
    random.seed(0)
    picks = Counter(rec.recommend_next(1, 10, 30.0) for _ in range(200))
    assert set(picks).issubset({100, 101, 102, 103})
    assert len(picks) > 1


def test_falls_back_when_no_candidates(fake_redis, push_history):
    class StaticFallback:
        def recommend_next(self, u, p, t):
            return 999

    push_history(user=1, events=[(10, 30.0)])
    rec = ExplorationRecommender(
        listen_history_redis=fake_redis,
        i2i_redis=fake_redis,
        fallback=StaticFallback(),
        epsilon=0.5,
        top_k=5,
        seed=0,
    )
    assert rec.recommend_next(1, 10, 30.0) == 999


def test_skips_seen_tracks(fake_redis, push_history, set_sasrec_recs):
    push_history(user=1, events=[(10, 30.0), (100, 40.0)])
    set_sasrec_recs(10, [100, 101, 102])
    rec = ExplorationRecommender(
        listen_history_redis=fake_redis,
        i2i_redis=fake_redis,
        fallback=None,
        epsilon=0.0,
        top_k=3,
        seed=0,
    )
    assert rec.recommend_next(1, 10, 30.0) == 101
