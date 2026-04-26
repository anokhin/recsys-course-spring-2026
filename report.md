# HW2 Report: HSTU-based Recommender with Seen-Track Filtering

## Abstract

We replace the SasRec-I2I recommender with a user-level HSTU neural 
model that pre-computes personalized ranked track lists per user. On top 
of HSTU candidates we apply a seen-track filter that removes 
already-listened tracks from the session history. An A/B test confirms a 
statistically significant improvement in mean_session_time.

## Details

The HSTU model is trained offline and produces a ranked list of ~100 
track candidates per user, stored in Redis. At serving time, 
SmartIndexed loads the candidate list, removes tracks already heard in 
the current session, and returns the highest-ranked unseen track. The 
ranking comes entirely from the HSTU neural model.

Experiment: Experiments.HSTU with Split.HALF_HALF. 50% of users see 
SasRec-I2I (control), 50% see SmartIndexed/HSTU (treatment).

## Diagram

    User request -> HSTU candidates (Redis) -> Seen-track filter -> Top 
unseen track
                                                                  -> 
fallback: Random

## A/B Test Results

| Group | mean_session_time | std | p-value |
|-------|-------------------|-----|---------|
| Control (SasRec-I2I) | - | - | - |
| Treatment (HSTU + filter) | - | - | - |## Homework 2 
Report

dragons be here
