import json
import pickle

import joblib
import numpy as np
import pandas as pd

from .recommender import Recommender


class MLRanker(Recommender):
    def __init__(
        self,
        model_path,
        recommendations_sasrec_redis,
        tracks_redis,
        listen_history_redis,
        baseline_recommender,
        fallback,
        topk=20,
        min_prev_time=0.80,
        abs_threshold=0.78,
        margin=0.10,
    ):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_cols = bundle["feature_cols"]

        self.recommendations_sasrec_redis = recommendations_sasrec_redis
        self.tracks_redis = tracks_redis
        self.listen_history_redis = listen_history_redis
        self.baseline_recommender = baseline_recommender
        self.fallback = fallback

        self.topk = int(topk)
        self.min_prev_time = float(min_prev_time)
        self.abs_threshold = float(abs_threshold)
        self.margin = float(margin)

    def _get_track_info(self, track_id):
        raw = self.tracks_redis.get(track_id)
        if raw is None:
            return None
        try:
            return pickle.loads(raw)
        except Exception:
            return None

    def _get_sasrec_candidates(self, prev_track: int):
        raw = self.recommendations_sasrec_redis.get(prev_track)
        if raw is None:
            return []
        try:
            return list(pickle.loads(raw))
        except Exception:
            return []

    def _get_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_history = self.listen_history_redis.lrange(key, 0, -1)
        history_tracks = []
        history_set = set()

        for item in raw_history:
            try:
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                record = json.loads(item)
                track = int(record["track"])
                history_tracks.append(track)
                history_set.add(track)
            except Exception:
                continue

        return history_tracks, history_set

    def _build_features(self, prev_track: int, recommendation: int, prev_time: float):
        prev_info = self._get_track_info(prev_track)
        rec_info = self._get_track_info(recommendation)
        if prev_info is None or rec_info is None:
            return None

        sasrec_list = self._get_sasrec_candidates(prev_track)
        if recommendation in sasrec_list:
            rank = sasrec_list.index(recommendation) + 1
            rr = 1.0 / rank
        else:
            rank = 0
            rr = 0.0

        return {
            "prev_track": int(prev_track),
            "recommendation": int(recommendation),
            "prev_time": float(prev_time),
            "same_artist": int(prev_info.artist == rec_info.artist),
            "sasrec_rank_from_prev": rank,
            "sasrec_rr_from_prev": rr,
            "sasrec_candidate_count": len(sasrec_list),
        }

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        baseline = None
        try:
            # 真 baseline：默认和 control 完全一致
            baseline = int(
                self.baseline_recommender.recommend_next(user, prev_track, prev_track_time)
            )

            try:
                prev_time = float(prev_track_time)
            except Exception:
                return baseline

            # 用户上一首听得不够深，不做覆盖
            if prev_time < self.min_prev_time:
                return baseline

            _, user_history_set = self._get_user_history(user)

            # 只在 SasRec 候选池内挑可替换项
            candidates = self._get_sasrec_candidates(prev_track)[: self.topk]
            filtered = []
            seen = set()

            for cand in candidates:
                try:
                    cand = int(cand)
                except Exception:
                    continue

                if cand == int(prev_track):
                    continue
                if cand in seen:
                    continue
                if cand in user_history_set:
                    continue

                seen.add(cand)
                filtered.append(cand)

            # baseline 自己如果不在 filtered，也强行补进去用于对比打分
            if baseline not in seen and baseline != int(prev_track):
                filtered.insert(0, baseline)

            if not filtered:
                return baseline

            rows = []
            valid_candidates = []
            for cand in filtered:
                feats = self._build_features(prev_track, cand, prev_time)
                if feats is None:
                    continue
                rows.append(feats)
                valid_candidates.append(int(cand))

            if not rows:
                return baseline

            X = pd.DataFrame(rows)
            for col in self.feature_cols:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_cols]

            scores = self.model.predict(X)
            if len(scores) != len(valid_candidates):
                return baseline

            score_map = {cand: float(score) for cand, score in zip(valid_candidates, scores)}

            baseline_score = score_map.get(baseline, None)
            if baseline_score is None:
                return baseline

            best_idx = int(np.argmax(scores))
            best_candidate = int(valid_candidates[best_idx])
            best_score = float(scores[best_idx])

            # 只有在非常高置信度时才允许覆盖 control 输出
            if (
                best_candidate != baseline
                and best_score >= self.abs_threshold
                and best_score >= baseline_score + self.margin
            ):
                return best_candidate

            return baseline

        except Exception:
            if baseline is not None:
                return int(baseline)
            return int(self.fallback.recommend_next(user, prev_track, prev_track_time))
