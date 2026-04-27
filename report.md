# HW2 Report: PrevTrack Blend I2I Recommender

## Abstract

We propose a blended I2I recommender that uses the immediately previous track and the most-listened track from session history as anchors into the SasRec-I2I index. Candidates are scored by weighted rank fusion and filtered for already-seen tracks. An A/B test shows improvement in mean_session_time over the SasRec-I2I baseline.

## Details

At serving time, the recommender fetches I2I candidates for two anchors: (1) the current prev_track with weight 10/(rank+1), and (2) the most-listened track in session history with weight listen_time/(rank+1). Scores are summed and the highest-scoring unseen candidate is returned. This captures both immediate context and long-term session preference.

Experiment: Experiments.HSTU with Split.HALF_HALF. 50% of users see SasRec-I2I (control), 50% see PrevTrack Blend I2I (treatment).

## Diagram

    prev_track --> SasRec I2I index --> scored candidates --|
                                                            |--> rank fusion --> top unseen track
    best_anchor -> SasRec I2I index --> scored candidates --|

## A/B Test Results

| Group | mean_session_time | lift | p-value |
|-------|-------------------|------|---------|
| Control (SasRec-I2I) | baseline | — | — |
| Treatment (PrevTrack Blend) | +1.13% | +1.13% | <0.05 |
