import json

import numpy as np
import pandas as pd


MISSING_RANK = 0


class LogisticResidualReranker:
    """
    Session-aware Logistic Regression residual reranker.

    SasRec-I2I is used as the default baseline policy.
    The logistic model only replaces the baseline when it predicts a sufficiently
    better candidate from the reconstructed candidate pool.
    """

    def __init__(
        self,
        model_path,
        listen_history_redis,
        hstu_redis,
        sasrec_redis,
        lfm_redis,
        catalog,
        baseline_recommender,
        fallback_recommender,
        top_k=20,
        min_prev_time=0.75,
        advantage_margin=0.10,
        min_best_score=0.55,
    ):
        with open(model_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        self.features = bundle["features"]
        self.scaler_mean = np.array(bundle["scaler_mean"], dtype=float)
        self.scaler_scale = np.array(bundle["scaler_scale"], dtype=float)
        self.coef = np.array(bundle["coef"], dtype=float)
        self.intercept = float(bundle["intercept"])

        self.listen_history_redis = listen_history_redis
        self.hstu_redis = hstu_redis
        self.sasrec_redis = sasrec_redis
        self.lfm_redis = lfm_redis
        self.catalog = catalog

        self.baseline_recommender = baseline_recommender
        self.fallback_recommender = fallback_recommender

        self.top_k = top_k
        self.min_prev_time = min_prev_time
        self.advantage_margin = advantage_margin
        self.min_best_score = min_best_score

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        baseline = self.baseline_recommender.recommend_next(
            user,
            prev_track,
            prev_track_time,
        )

        if baseline is None:
            return self.fallback_recommender.recommend_next(
                user,
                prev_track,
                prev_track_time,
            )

        baseline = int(baseline)

        if prev_track_time < self.min_prev_time:
            return baseline

        history = self._load_history(user)

        candidate_info = self._build_candidate_info(user, prev_track)

        if baseline not in candidate_info:
            candidate_info[baseline] = {
                "sasrec_rank": 1,
                "hstu_rank": MISSING_RANK,
                "lfm_rank": MISSING_RANK,
            }

        candidates = list(candidate_info.keys())

        if not candidates:
            return baseline

        X = self._make_features(
            prev_track_time=prev_track_time,
            candidates=candidates,
            candidate_info=candidate_info,
            history=history,
        )

        X_values = X[self.features].values.astype(float)
        X_scaled = (X_values - self.scaler_mean) / self.scaler_scale
        logits = X_scaled @ self.coef + self.intercept
        scores = 1.0 / (1.0 + np.exp(-logits))

        candidate_to_score = {
            int(candidate): float(score)
            for candidate, score in zip(candidates, scores)
        }

        baseline_score = candidate_to_score.get(baseline, 0.0)

        best_candidate = max(
            candidates,
            key=lambda c: candidate_to_score.get(int(c), 0.0),
        )
        best_candidate = int(best_candidate)
        best_score = candidate_to_score.get(best_candidate, 0.0)

        if (
            best_candidate != baseline
            and best_score >= self.min_best_score
            and best_score >= baseline_score + self.advantage_margin
        ):
            return best_candidate

        return baseline

    def _load_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)

        history = []

        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            try:
                obj = json.loads(raw)
                history.append(
                    {
                        "track": int(obj["track"]),
                        "time": float(obj["time"]),
                    }
                )
            except Exception:
                continue

        return history

    # def _build_candidate_info(self, user: int, prev_track: int):
    #         info = {}

    #         def add(track, source, rank):
    #             try:
    #                 track = int(track)
    #             except Exception:
    #                 return

    #             rank = int(rank) + 1

    #             if track not in info:
    #                 info[track] = {
    #                     "sasrec_rank": MISSING_RANK,
    #                     "hstu_rank": MISSING_RANK,
    #                     "lfm_rank": MISSING_RANK,
    #                 }

    #             field = f"{source}_rank"
    #             old = info[track].get(field, MISSING_RANK)

    #             if old == MISSING_RANK:
    #                 info[track][field] = rank
    #             else:
    #                 info[track][field] = min(old, rank)

    #         sasrec_recs = self._load_i2i(self.sasrec_redis, prev_track, self.top_k)
    #         for rank, track in enumerate(sasrec_recs):
    #             add(track, "sasrec", rank)

    #         raw = self.hstu_redis.get(user)
    #         if raw is not None:
    #             try:
    #                 hstu_recs = list(self.catalog.from_bytes(raw))
    #                 for rank, track in enumerate(hstu_recs[: self.top_k]):
    #                     add(track, "hstu", rank)
    #             except Exception:
    #                 pass

    #         lfm_recs = self._load_i2i(self.lfm_redis, prev_track, self.top_k)
    #         for rank, track in enumerate(lfm_recs):
    #             add(track, "lfm", rank)

    #         return info

    def _build_candidate_info(self, user: int, prev_track: int):
        info = {}

        def add(track, source, rank):
            try:
                track = int(track)
            except Exception:
                return

            rank = int(rank) + 1

            if track not in info:
                info[track] = {
                    "sasrec_rank": MISSING_RANK,
                    "hstu_rank": MISSING_RANK,
                    "lfm_rank": MISSING_RANK,
                }

            field = f"{source}_rank"
            old = info[track].get(field, MISSING_RANK)

            if old == MISSING_RANK:
                info[track][field] = rank
            else:
                info[track][field] = min(old, rank)

        # 只使用 SasRec-I2I 候选
        sasrec_recs = self._load_i2i(self.sasrec_redis, prev_track, self.top_k)
        for rank, track in enumerate(sasrec_recs):
            add(track, "sasrec", rank)

        return info

    def _load_i2i(self, redis_conn, track: int, k: int):
        raw = redis_conn.get(track)

        if raw is None:
            return []

        try:
            recs = list(self.catalog.from_bytes(raw))
            return [int(x) for x in recs[:k]]
        except Exception:
            return []

    def _make_features(
            self,
            prev_track_time: float,
            candidates,
            candidate_info,
            history,
        ):
            recent = history[:5]
            recent_times = np.array([x["time"] for x in recent], dtype=float)

            hist_len = len(recent)
            recent_avg_time = float(np.mean(recent_times)) if hist_len else 0.0
            recent_good_frac = float(np.mean(recent_times >= 0.70)) if hist_len else 0.0
            recent_skip_frac = float(np.mean(recent_times <= 0.20)) if hist_len else 0.0

            rows = []

            for c in candidates:
                info = candidate_info.get(
                    int(c),
                    {
                        "sasrec_rank": MISSING_RANK,
                        "hstu_rank": MISSING_RANK,
                        "lfm_rank": MISSING_RANK,
                    },
                )

                source_count = int(info.get("sasrec_rank", 0) > 0)
                source_count += int(info.get("hstu_rank", 0) > 0)
                source_count += int(info.get("lfm_rank", 0) > 0)

                rows.append(
                    {
                        "prev_time": float(prev_track_time),
                        "hist_len": hist_len,
                        "recent_avg_time": recent_avg_time,
                        "recent_last_time": float(prev_track_time),
                        "recent_good_frac": recent_good_frac,
                        "recent_skip_frac": recent_skip_frac,
                        "sasrec_rank": int(info.get("sasrec_rank", MISSING_RANK)),
                        "hstu_rank": int(info.get("hstu_rank", MISSING_RANK)),
                        "lfm_rank": int(info.get("lfm_rank", MISSING_RANK)),
                        "source_count": source_count,
                    }
                )

            return pd.DataFrame(rows)