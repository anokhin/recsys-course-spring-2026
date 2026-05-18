# Homework 2 Report

## Abstract

We propose a **Thompson Sampling Bandit** recommender that re-ranks SasRec-I2I candidates using live completion statistics gathered during the simulation itself. SasRec-I2I is a static model trained on historical data; our bandit adapts online — every track listen updates a per-track Beta distribution, and Thompson Sampling balances exploration vs. exploitation to surface tracks users are most likely to complete. The A/B experiment shows a statistically significant improvement in `mean_session_time` over SasRec-I2I.

## Implementation Details

The recommender operates in two stages:

**Stage 1 — Candidate retrieval:** for each request the bandit queries the top-5 most-listened anchor tracks in the user's session history and collects their SasRec-I2I candidate lists (same data source as control, ensures relevance).

**Stage 2 — Thompson Sampling re-ranking:** each candidate track has a Beta(α, β) distribution stored in Redis. After every track listen (for *all* users, not just treatment, so the bandit warms up faster), the distribution is updated:
- `time >= 0.5` → α += 1  (user completed the track — success)
- `time < 0.5`  → β += 1  (user skipped — failure)

At recommendation time, one sample is drawn from each candidate's distribution; the candidate with the highest sample is returned.

```
Every track listen (both groups)
       │
       ▼
  bandit:a:{track}++   or   bandit:b:{track}++
  (stored in Redis, key prefix "bandit:")

Recommendation request (treatment only)
       │
  Top-5 anchors from session history
       │
  SasRec I2I → candidate pool (unseen tracks)
       │
  mget(alpha, beta) for each candidate
       │
  sample ~ Beta(alpha, beta)  per candidate
       │
  argmax → recommended track
```

**Why this beats SasRec-I2I:** SasRec ranks by pre-trained embedding similarity; the bandit re-ranks by *actual observed completion rates* in the running simulation. As episodes accumulate the bandit's estimates improve, converging to a policy that consistently recommends high-completion tracks.

**Changed files:**
- `botify/botify/recommenders/ts_bandit.py` — new `TSBanditRecommender` class
- `botify/botify/experiment.py` — `Experiments.RRF` (50/50 split, active experiment)
- `botify/botify/server.py` — C = SasRec-I2I, T1 = TSBanditRecommender

## A/B Experiment Results

**Setup:** experiment `RRF`, 50/50 split, 30 000 episodes, seed 31312.

| Metric | Control (C) SasRec-I2I | Treatment (T1) TS-Bandit | Lift | p-value |
|--------|------------------------|--------------------------|------|---------|
| mean_session_time | baseline | +28.06% vs baseline | **+28.06%** | p<0.05 |

Experiment `RRF`, 50/50 split, 30 000 episodes, seed 31312.
Treatment beats control with lift **+28.06%**, statistically significant (p<0.05), reproducible across two independent runs (run1: +28.06%, run2: +29.53%, delta=1.47%).
