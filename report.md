# Homework 2: Music Track Recommender

## Abstract

I propose a two-stage ML-based recommender for Botify. The recommender first generates a pool of candidate tracks using several independent sources: global popularity, user listening history and co-visitation statistics from historical sessions. Then a supervised machine learning model reranks these candidates using user-track, track-level and session-level features. Unlike the SasRec-I2I baseline, the proposed solution uses an explicit ML ranking stage trained on collected interaction logs. The goal of the experiment is to increase mean_session_time in a fair A/B test against SasRec-I2I.

## Details

The recommender consists of two stages. At the first stage, candidate tracks are collected from popular tracks, recently listened tracks and co-visited tracks from historical sessions. This gives the system both stable fallback recommendations and personalized candidates related to the current user context.

At the second stage, candidates are scored by a RandomForestClassifier. The model uses features such as track popularity, average listening time, number of previous user-track interactions, co-visitation score with recent user history, user history length and whether the user has already listened to the track. The model is trained on positive examples from real session continuations and negative examples sampled from popular tracks not present in the same session.

```mermaid
flowchart LR
    Logs[Interaction logs] --> Candidates[Candidate generation]
    Logs --> Features[Feature engineering]
    Candidates --> Ranker[ML ranking model]
    Features --> Ranker
    Ranker --> Recs[Final recommendations]
    Recs --> AB[AB test]
