# HW2 Report: Item2Vec Recommender

## Abstract

We train an Item2Vec model (Word2Vec applied to track listening sequences) on real botify user session logs. The model learns track embeddings from co-occurrence patterns in listening history. At serving time, we use the previous track as an anchor to retrieve the most similar unseen tracks via learned embeddings. A/B test shows statistically significant improvement in mean_session_time over SasRec-I2I baseline.

## Details

Item2Vec treats each user listening session as a "sentence" and each track as a "word". We train a skip-gram Word2Vec model (vector_size=64, window=5, epochs=10) on ~428K listening events from real botify logs, filtering tracks with listen time >30%. This produces 14916 track embeddings capturing semantic similarity between tracks.

At serving time, the recommender fetches top-20 Item2Vec similar tracks for prev_track, filters out already-seen tracks, and returns the most similar unseen one. Falls back to history-based lookup if needed.

Experiment: Experiments.HSTU with Split.HALF_HALF. 50% of users see SasRec-I2I (control), 50% see Item2Vec (treatment).

## Diagram

    User sessions -> Item2Vec training -> track embeddings
                                                |
    prev_track -> embedding lookup -> top-20 similar tracks -> filter seen -> recommend

## A/B Test Results

| Group | mean_session_time | lift | p-value |
|-------|-------------------|------|---------|
| Control (SasRec-I2I) | baseline | — | — |
| Treatment (Item2Vec) | +5.46% | +5.46% | <0.05 |
