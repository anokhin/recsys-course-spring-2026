# Homework 2 Report — LightGBM LambdaRank with HSTU-feature gating

## Abstract

We replace the static, user-keyed HSTU recommender with a **two-stage ML
ranker**: a multi-source candidate generator anchored on the user's recent
listen history, followed by a **LightGBM LambdaRank** model that re-orders
the pool. The model is trained offline on the `experiments.HSTU == "C"`
control logs of a previous SasRec-I2I run with a learning-to-rank objective
(positive = the candidate the user actually accepted with `time ≥ 0.7`,
negatives = sibling candidates from the same state). At serving time the
ranker is wrapped by a confidence gate: it only overrides the SasRec-I2I
baseline pick when its preferred candidate beats baseline by a clear margin,
keeping the asymmetric risk profile (small cost when the model is wrong,
big win when it's right). The HSTU pre-computed user top-N is retained as
**features** (`hstu_rank_inv`, `hstu_present`), turning the offline asset
into a ranking signal rather than a candidate list.

## Details

**Candidate generation.** For each request we read the user's last 10 plays
from Redis, take the most recent 4 as anchors, and pull the top-10 I2I
neighbours of each anchor from two sources: SasRec-I2I and LightFM-I2I. We
deduplicate against the user's seen set and the current `prev_track`,
yielding ≤ 50 candidates per request. We always inject the SasRec-I2I
baseline pick into the pool so its score is directly comparable.

**Feature extraction.** For every (state, candidate) pair we compute 24
features in four buckets:

- **Session state**: `hist_len`, `avg_time`, `last_time`, `good_frac`
  (fraction of recent plays with `time ≥ 0.7`), `skip_frac`, `unique_artists`.
- **Content match against the user's likes**: `same_artist_last`,
  `cand_artist_repeat`, `genre_jaccard_liked` (Jaccard over genres of
  recent good plays), `mood_match_count`, `year_dist`, `artist_fans_log`.
- **Cross-source ranking**: per anchor, `sasrec_hits / lfm_hits` (how many
  anchors retrieved this candidate), `*_best_rr` (best `1/rank`),
  `*_weighted_rr` (rank inverse weighted by the anchor's listen-time),
  and `source_agreement` (how often both sources retrieve it together).
- **HSTU + popularity** (the unique-to-us signals): `hstu_rank_inv`,
  `hstu_present` (where does the candidate sit in the user's pre-computed
  HSTU top-100), and global stats from logs: `cand_global_mean_time`,
  `cand_global_good_rate`, `cand_global_log_count`.

**Model.** A LightGBM `lambdarank` booster is trained on 60,120 control
states (≈ 2.3 M (state, candidate) rows) with one query per state and
binary relevance (1 for the true accepted candidate, 0 otherwise). With
24 features, 31 leaves and early stopping, validation **NDCG@1 = 0.636**
on a held-out 15% group split — the model picks the right next track from
~38 candidates 64% of the time. The booster is exported to a 320 KB text
dump and loaded directly into the Flask service; per-request inference is
one `numpy` matmul over 50 rows.

**Confidence gate (serving-time decision rule).** The ranker is wrapped so
that the SasRec-I2I baseline is the default response. We only override
when the model strictly prefers a *different* candidate **and** its score
beats the baseline candidate's score by ≥ 0.05 **and** the candidate's
artist hasn't already filled the last three plays. If `prev_track_time`
is below 0.65 we skip the model entirely — the user has already
disengaged and we have nothing to add.

```text
                       OFFLINE TRAINING
   Botify control logs (HSTU=="C")  ──►  state, candidate, label
                                              │
                                              ▼
                                  LightGBM lambdarank
                                              │
                                              ▼
                       botify/data/learned_ranker.lgb (+ meta.json)

                       ONLINE SERVING (T1)
   POST /next/{user}
        │
        ▼
   user history (Redis)
        │
        ▼   anchors = last-4 plays
   pool = SasRec-I2I  ∪  LightFM-I2I  (top-10 per anchor)  + baseline
        │
        ▼
   features per candidate (HSTU rank, I2I cross-source, content, popularity)
        │
        ▼
   LightGBM scores  ──►  argmax + confidence gate vs baseline
        │
        ▼
   override ?  best_cand  :  SasRec-I2I baseline
```

## A/B Experiment Results

Experiment: `HW2_RANKER` (50/50 split). Control `C` = `sasrec_i2i_recommender`,
treatment `T1` = `learned_gate_ranker`. Run with `EPISODES=30000`,
`SEED=31312`.

| treatment | metric                  | effect_pct | lower_pct | upper_pct | control_mean | treatment_mean | significant |
|-----------|-------------------------|-----------:|----------:|----------:|-------------:|---------------:|-------------|
| T1        | **mean_time_per_session** | **+24.55** | +22.43    | +26.68    | 7.0402       | 8.7687         | **True**    |
| T1        | mean_tracks_per_session | +14.44     | +13.12    | +15.77    | 12.0249      | 13.7617        | True        |
| T1        | time                    | +25.59     | +22.44    | +28.74    | 22.0365      | 27.6760        | True        |
| T1        | sessions                | +0.18      | −1.91     | +2.28     | 3.1918       | 3.1977         | False       |
| T1        | mean_request_latency    | +115.48    | +111.81   | +119.15   | 1.1553       | 2.4895         | True        |

The target metric `mean_time_per_session` improves by **+24.6%** with a
99% CI strictly above zero. `mean_tracks_per_session` rises +14.4%, so
listeners both stay longer per session **and** consume more tracks before
abandoning — the ranker is keeping users in the same musical neighbourhood
where they are clearly engaged. `sessions` per user is unchanged
(+0.18%, not significant), so we are not just splitting longer sessions
into more shorter ones. Latency rises from 1.16 ms to 2.49 ms — well
within budget and ≪ the 8 ms a previous candidate generator
(`HSTUSessionRecommender`) was paying — because the ranker reuses the
same Redis I2I lookups and runs a single 50×24 LightGBM forward per
request.
