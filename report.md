# HW2 Report: Item2Vec Aggregated Recommender

## Abstract

We train an Item2Vec model (Word2Vec applied to track listening sequences) on real botify user session logs. The model learns track embeddings from co-occurrence in listening history. At serving time, we aggregate Item2Vec candidates across the full session history weighted by listen time, scoring each candidate by rank-weighted sum. A/B test shows statistically significant improvement in mean_session_time over SasRec-I2I baseline.

## Details

Item2Vec treats each user session as a sentence and each track as a word. We train skip-gram Word2Vec (vector_size=64, window=5, epochs=10) on ~428K listening events from real botify logs, filtering tracks with listen time >30%. This produces 14916 track embeddings capturing semantic similarity between tracks.

At serving time, Item2VecAgg fetches candidates for prev_track (weight=10) and all history anchors (weight=listen_time), scores by weight/(rank+1) and returns the highest scored unseen track.

Experiment: Experiments.HSTU with Split.HALF_HALF. 50% see SasRec-I2I (control), 50% see Item2VecAgg (treatment).

## Diagram

    User sessions -> Item2Vec training -> track embeddings
                                                |
    prev_track + history -> embedding lookup -> scored candidates -> top unseen track

## A/B Test Results

| Group | mean_session_time | lift | p-value |
|-------|-------------------|------|---------|
| Control (SasRec-I2I) | baseline | — | — |
| Treatment (Item2VecAgg) | +2.84% | +2.84% | <0.05 |
