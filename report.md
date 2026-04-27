# HW2 Report: PrevTrack I2I Recommender

## Abstract

We propose a recommender that uses the immediately previous track as the primary anchor for SasRec I2I lookup, combined with session history fallback ordered by recency. This captures the most recent user intent signal rather than randomly sampling from history anchors as in the baseline SasRec-I2I.

## Details

At serving time, the recommender first fetches I2I candidates for prev_track from the SasRec neural model index. If no unseen candidates found, it falls back to history anchors ordered by recency. SasRec is a self-attentive sequential recommendation transformer model trained on user listening history.

Experiment: Experiments.HSTU with Split.HALF_HALF. 50% of users see SasRec-I2I (control), 50% see PrevTrack I2I (treatment).

## Diagram

    prev_track -> SasRec I2I index -> first unseen candidate -> recommended track
                                              |
                                    (if empty) fallback:
                                    history tracks by recency -> SasRec I2I index

## A/B Test Results

| Group | mean_session_time | lift | p-value |
|-------|-------------------|------|---------|
| Control (SasRec-I2I) | baseline | — | — |
| Treatment (PrevTrack I2I) | +1.54% | +1.54% | 0.07 |
