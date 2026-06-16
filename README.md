# Homework 2 Report

## Abstract

The goal of this homework was to improve the Botify recommendation service and outperform the baseline `SasRec-I2I` in an honest online A/B test using `mean_time_per_session` as the target metric.

My final solution is a **session-aware online ML ranker** with a **conservative gating policy**. The control group uses the original `SasRec-I2I` without modifications. The treatment group builds a small candidate set from multiple item-to-item sources and applies a trained ML model to rerank them using short-term session context. The model is only allowed to override the baseline recommendation when its confidence is sufficiently high. This makes the system more stable and reduces harmful aggressive replacements.

According to the automatic evaluation, the final solution achieved a statistically significant improvement over control and passed reproducibility checks.

---

## Method

### 1. Control

The control group is the original `SasRec-I2I` recommender.

### 2. Treatment

The treatment group uses an online recommender implemented in:

- `botify/botify/recommenders/session_gate_ranker.py`

Its logic is the following:

1. Read the recent user listening history from Redis.
2. Build a candidate pool from:
   - `SasRec-I2I`
   - `LightFM-I2I`
3. Compute session-aware features for every candidate.
4. Score candidates with a trained logistic regression model.
5. Compare the best ML candidate with the baseline recommendation.
6. Override the baseline only if several conservative gating conditions are satisfied.

This is not a heuristic replacement table. The final decision in treatment is made by an ML model working online on top of the current user session.

---

## Feature design

The model uses features that describe both the current session state and the candidate quality.

### Session features

- history length
- average recent listening time
- listening time of the previous track
- fraction of “good” recent listens
- fraction of recent skips

### Candidate–history interaction features

- whether candidate artist matches the previous track artist
- whether candidate title prefix is similar to previous track title
- how often candidate artist appeared recently

### Retrieval-source features

- number of times the candidate appears in recent `SasRec` neighbors
- number of times the candidate appears in recent `LightFM` neighbors
- weighted inverse-rank signals from both sources
- agreement between retrieval sources

These features are designed to capture whether a candidate is suitable for the current short-term session context, instead of only predicting the next track ID.

---

## Training

The training script is:

- `botify/train_session_gate_ranker.py`

Training data comes from logged Botify interactions collected from the simulator through the service itself.

Only **control-group logs** are used for training. For every session state, a candidate set is built from `SasRec` and `LightFM`. A positive example is defined as the true next track in cases where the next listening time is sufficiently high. Other candidates from the same candidate pool are treated as negatives.

The model is trained with:

- `StandardScaler`
- `LogisticRegression`

User-based splitting is used during validation to reduce leakage between train and validation parts.

The trained model is stored in:

- `botify/session_gate_ranker_bundle.joblib`

---

## Conservative gating

A key part of the final solution is that treatment does **not** blindly replace the baseline.

The model is allowed to override the baseline only when:

- the previous listening time is high enough,
- the predicted score is above an absolute threshold,
- the predicted score is better than the baseline score by a margin,
- the replacement does not create an overly repetitive artist pattern.

If these conditions are not satisfied, the system returns the original baseline recommendation.

This conservative policy turned out to be important for stability and reproducibility.

---

## Online integration

The online service was modified in:

- `botify/botify/server.py`
- `botify/botify/config.json`

The Docker environment was updated so the recommender can load the model and required dependencies correctly:

- `botify/Dockerfile`
- `botify/requirements.txt`

---

## Results

Automatic evaluation produced the following result:

- Run 1 effect: **+19.9%**
- Run 2 effect: **+21.76%**
- Reproducibility delta: **1.86%** (threshold 10%)

Final A/B result:

- Control beaten: **yes**
- Significant (`p < 0.05`): **yes**
- Lift (`mean_time_per_session`): **+19.9%**

Automatic score:

- **35 / 35**

Thus, the final method successfully improves the target metric and remains reproducible across repeated runs.

---

## Conclusion

The final solution is an **online session-aware ML reranker with conservative gating**. Unlike static recommendation files, it uses the current short-term session context, combines multiple retrieval sources, and only overrides the baseline in high-confidence situations.

This approach achieved a statistically significant and reproducible improvement over `SasRec-I2I` on the target metric `mean_time_per_session`, while remaining simple enough to serve online inside Botify.

