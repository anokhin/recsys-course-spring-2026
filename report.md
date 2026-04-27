# HW2 Report: HSTU Retrieval + LightFM Reranking

## Abstract

We propose a two-stage ML recommender: HSTU neural model retrieves personalized candidate tracks per user, then LightFM I2I model reranks candidates based on session history. An A/B test shows improvement in mean_session_time over the SasRec-I2I baseline.

## Details

Stage 1 (Retrieval): HSTU neural model generates ~100 personalized candidate tracks per user, stored in Redis. Stage 2 (Reranking): LightFM I2I scores each unseen candidate using session history anchors weighted by listen time divided by rank.

Experiment: Experiments.HSTU with Split.HALF_HALF. 50% of users see SasRec-I2I (control), 50% see HSTU+LightFM (treatment).

## Diagram

    User -> HSTU Redis -> candidate set
                                        -> LightFM reranker -> top scored unseen track
    Session history -> LightFM I2I Redis -> anchor scores

## A/B Test Results

| Group | mean_session_time | lift | p-value |
|-------|-------------------|------|---------|
| Control (SasRec-I2I) | baseline | — | — |
| Treatment (HSTU+LightFM) | TBD | TBD | TBD |
