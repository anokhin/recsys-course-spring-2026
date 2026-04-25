# HW2 Report — Transition-based Item-Item Recommender

## Abstract

We replace `SasRec-I2I` with an item-item recommender whose similarities are
**learned directly from the sequence of user transitions** in bootstrap
simulator logs, rather than from a sequence-prediction loss. For each pair of
tracks `(a, b)` we accumulate the listened fraction of `b` whenever `b` was
played immediately after `a` within the same user session. This raw signal is
smoothed by adding a small weight `α = 0.05` of the cosine similarity of iALS
item factors (trained on the same logs) so that pairs with no observed
transition still receive a sensible score. Tracks that never appear as a
"prev_track" in the logs fall back to a 50/50 blend of iALS-factor cosine and
sentence-transformer content-embedding cosine. The resulting top-10 i2i list
per track is served through the **existing** `I2IRecommender`, exactly the
same serving path used by SasRec-I2I, so the only thing that changes between
arms is the learned item-item table.

## Details

```
sim run (seed=11111, 30k episodes, control = SasRec-I2I, treatment = HSTU)
        │
        ▼
data.json  ─── user×track listen_time ───────────┐
        │                                        │
        │   tracks.json ─── sentence-transformer ┤
        │   (summary | mood | genres |           │
        │    artist | country)                   │
        │                                        │
        │   user→track→listen_time ──── iALS ────┤
        │   (factors=64, iter=20)                │
        │                                        │
        │   raw events (timestamp, msg) ─────────┤
        │                                        │
        ▼                                        ▼
   transition matrix                   iALS-factor cosine
   T[a,b] = Σ listen_time(b) over     (used as smoothing for
   sessions where b followed a         cold pairs)
        │                                        │
        └─────────────── + 0.05 · ───────────────┘
                            │
                            ▼
                     score(a, b)
                            │
                            ▼
              top-10 per anchor → our_i2i.jsonl
                            │
                            ▼
        I2IRecommender(history, our_i2i, random_fallback)
                            │
                            ▼
   /next/<user>: pick anchor from last 10 listens by listen_time,
                 return first unseen candidate from anchor's top-10
```

The bootstrap data was collected with a different seed (`11111`) than the A/B
seed (`31312`) so training data and evaluation data are disjoint. The
training pipeline lives in `script/`: `embed_tracks.py` produces
`botify/data/track_embeddings.npy` (16198 × 384, all-MiniLM-L6-v2);
`train_recsys.py` loads logs via `load_logs.py`, fits iALS for cold-pair
smoothing, builds the transition matrix from raw events, blends them, and
writes `botify/data/our_i2i.jsonl` (16198 entries, top-10 each). At runtime
`botify/botify/server.py` loads the JSONL into Redis and routes 50/50 traffic
through `Experiments.OUR_MODEL`: control → SasRec-I2I, treatment → our
transition-based I2IRecommender. Tracks that appear only as targets of
transitions (never as anchors) fall back to a 50/50 iALS+content blend so
every item has a top-10.

## A/B Results

Control: SasRec-I2I. Treatment T1: our transition-based I2I. 50/50 split,
seed 31312, 30000 episodes.

| metric | control mean | treatment mean | effect | 95% CI | significant |
|---|---|---|---|---|---|
| mean_time_per_session | 6.9813 | 7.5751 | **+8.51%** | [+6.38%, +10.63%] | ✅ |
| mean_tracks_per_session | 11.9776 | 12.5741 | +4.98% | [+3.65%, +6.31%] | ✅ |
| time (total) | 21.3692 | 23.7891 | +11.32% | [+8.30%, +14.35%] | ✅ |
| sessions | 3.1273 | 3.1886 | +1.96% | [-0.12%, +4.04%] | ✗ |
| mean_request_latency | 0.6029 | 0.4799 | -20.40% | [-60.26%, +19.46%] | ✗ |

The treatment beats SasRec-I2I on the target metric `mean_time_per_session`
by 8.51% with the entire confidence interval above zero. Users in the
treatment arm both listen to more tracks per session and listen to a larger
fraction of each track. The latency drop is not significant; the session-count
metric is roughly flat as expected (we change recommendations, not session
arrival).
