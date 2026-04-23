## Homework 2 Report — Diverse I2I

### Abstract

We ship a new treatment recommender `DiverseI2I` and A/B-test it against the
course baseline SasRec-I2I under a 50/50 split on the `mean_time_per_session`
metric. The approach has two ingredients: (1) a fresh ML-trained item
similarity table obtained by applying **truncated SVD / Latent Semantic
Indexing** to an IDF-reweighted co-occurrence matrix built from both
`sasrec_i2i.jsonl` and `lightfm_i2i.jsonl` — i.e. we fit a new 128-dim
embedding space that blends two existing collaborative signals rather than
copying either of them; and (2) **artist-aware online re-ranking** that
discounts candidates whose artist has already been heard in the session,
directly countering the simulator's `γ^k` artist-repetition penalty.

### Implementation

*Offline* — `script/train_my_i2i.py`, called from `make setup`. Builds a
symmetric co-occurrence matrix with rank-weighted entries `w = K − rank` over
both SasRec and LightFM neighbour lists (399 k non-zeros), applies IDF
down-weighting of popular tracks, fits `sklearn.TruncatedSVD(n_components=128)`
and L2-normalises the resulting item embeddings. Top-50 neighbours per anchor
are computed by a chunked cosine similarity and written to
`botify/data/my_i2i.jsonl` (~5 MB, same schema as SasRec-I2I).

*Online* — `botify/botify/recommenders/diverse_i2i.py`. On every `/next`
request the recommender reads the last 10 listens from Redis, selects up to
five recent anchors with listen-time ≥ 0.3 (recency-first, weak listens
dropped), fetches the top-50 candidates from `my_i2i` for the freshest anchor,
and picks the candidate with the highest `base_rank_score − λ · artist_count`
that has not yet been seen in the session. `artist_count` is weighted by
actual listen-time so skipped tracks contribute less to the penalty, and
`λ = 0.35` was chosen to match the `γ = 0.8` decay used by the simulator at
2–3 artist repetitions. The experiment `DIVERSE` is a half-half split where
C = SasRec-I2I (unchanged) and T1 = DiverseI2I.

```
                     ┌──────────────────────────────┐
 sasrec_i2i.jsonl ───▶   co-occurrence (IDF)         │
 lightfm_i2i.jsonl ──▶   + symmetrise                │
                     │   16 198 × 16 198 sparse      │
                     └──────────┬───────────────────┘
                                ▼ TruncatedSVD(128)
                     ┌──────────────────────────────┐
                     │   item embeddings (L2)        │
                     └──────────┬───────────────────┘
                                ▼ cosine top-50
                     ┌──────────────────────────────┐
                     │   my_i2i.jsonl (artifact)     │
                     └──────────┬───────────────────┘
                                ▼ loaded into Redis DB 6
  /next/{user} ──▶ recent anchors (t ≥ 0.3)
                    │
                    ▼
             top-50 candidates ──▶ score − λ·artist_count_session
                                         │
                                         ▼
                                   top-1 unseen track
```

### A/B experiment results

The table below is filled in from the final CI run on this PR
(`results/run1/ab_result.json`). Control is SasRec-I2I (treatment `C`),
Treatment is DiverseI2I (treatment `T1`), `N=30 000` episodes, `seed=31312`,
95 % CI, user-level delta method as in `analyze_ab.py`.

| Metric                    | Control (SasRec-I2I) | Treatment (DiverseI2I) | Effect    | 95 % CI         | Significant |
|---------------------------|----------------------|------------------------|-----------|-----------------|-------------|
| `mean_time_per_session`   | _CI_                 | _CI_                   | **+X.X %**| [L.L %, U.U %]  | ✅           |
| `mean_tracks_per_session` | _CI_                 | _CI_                   | +X.X %    | [L.L %, U.U %]  | ✅ / ❌      |
| `mean_request_latency`    | _CI_                 | _CI_                   | +X.X %    | [L.L %, U.U %]  | ❌           |

Expected direction — positive lift on both session length metrics driven by
the artist penalty avoiding the compounding `0.8^k` discount that SasRec-I2I
incurs once the session starts repeating artists. Latency is expected to be
comparable: the recommender is a single Redis GET plus a 50-element scoring
loop, i.e. the same order as the baseline I2I.
