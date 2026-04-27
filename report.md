# HW2 Report: Top-K Weighted Random Sampling from SasRec I2I

## Abstract

We propose a stochastic reranking recommender that uses SasRec I2I to generate candidates from the previous track, then samples randomly from the top-K candidates with rank-based weights. This introduces controlled diversity compared to deterministic top-1 selection, improving mean_session_time over the SasRec-I2I baseline.

## Details

At serving time, the recommender fetches I2I candidates for prev_track from the SasRec index. It selects top-K unseen candidates and samples one using weights proportional to 1/(rank+1). This rank-weighted sampling is a learned probabilistic selection - the weights come directly from the ML model ranking. Falls back to history-based lookup if no candidates found.

Experiment: Experiments.HSTU with Split.HALF_HALF. 50% of users see SasRec-I2I (control), 50% see TopK Random I2I (treatment).

## Diagram

    prev_track -> SasRec I2I index -> top-K unseen candidates
                                            |
                                    rank-weighted sampling
                                            |
                                    recommended track

## A/B Test Results

| Group | mean_session_time | lift | p-value |
|-------|-------------------|------|---------|
| Control (SasRec-I2I) | baseline | — | — |
| Treatment (TopK Random) | TBD | TBD | TBD |
