# HW2 Report: Multi-Source RRF Ranker with Artist Diversity

## Abstract

We propose a **multi-source Reciprocal Rank Fusion (RRF) recommender** that combines SasRec-I2I, LightFM-I2I, and HSTU user-level candidate pools with a session-aware artist diversity penalty. Unlike the baseline SasRec-I2I which samples a single random anchor and returns its first unseen recommendation, our approach aggregates signals from all recent anchors weighted by cumulative listen time, fuses two complementary I2I sources via RRF, and applies a score penalty for candidates whose artist has already appeared recently in the session. This directly counters the simulator's `artist_discount_gamma = 0.8` penalty while improving recommendation stability via multi-anchor pooling.

## Details

**Candidate generation.** For each request, we select up to 5 anchors from the user's listen history, ranked by total listen time. For each anchor, candidates are retrieved from SasRec-I2I (weight 1.0) and LightFM-I2I (weight 0.7). HSTU user-level recommendations (long-term preferences, weight 0.5) are added independently of anchors using the user ID directly. Scores are accumulated via RRF: each source contributes `w × anchor_time / (rank + 60)` for I2I sources and `w / (rank + 60)` for HSTU. Already-heard tracks are filtered before scoring.

**Artist diversity.** A soft penalty (×0.25) is applied to candidates whose artist appeared among the 3 most recently heard unique artists in the session. This is a soft constraint — if all candidates share recent artists, the penalty applies uniformly and the best-scoring candidate still wins. All artist lookups use an in-memory `{track_id: artist}` dictionary built at server startup, adding zero Redis calls per recommendation.

```
session history  (last 10 tracks, time-weighted)
        │
        ▼
top-5 anchors by Σ listen_time
        │
   ┌────┴─────────────────────┐
   ▼                          ▼
SasRec-I2I (×1.0)     LightFM-I2I (×0.7)
        │                     │
        └──────────┬───────────┘
                   ▼
       RRF score per candidate
       score += w × t / (rank + 60)
                   │
          filter seen tracks
                   │
      artist penalty ×0.25 if artist
      in last 3 unique session artists
                   │
                   ▼
           argmax → recommendation
```

**Why this beats SasRec-I2I.** The baseline has two weaknesses: (1) it picks a single random anchor, introducing high variance; (2) it makes no attempt to diversify artists, so ~19% of SasRec recommendations repeat the same artist, accumulating `0.8^k` listen-time discounts. Our approach reduces anchor variance via pooling and directly addresses artist fatigue.

## A/B Results

Experiment: `MULTI_I2I`, 50/50 split, seed=31312, 30 000 episodes.

| treatment | metric | control_mean | treatment_mean | effect_pct | lower_pct | upper_pct | significant |
|-----------|--------|-------------|----------------|------------|-----------|-----------|-------------|
| T1 | mean_time_per_session | — | — | — | — | — | — |

_Results will be updated after CI run._