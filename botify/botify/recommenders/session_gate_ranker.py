import json
import math
import pickle
from collections import Counter
from typing import Dict, List, Tuple
import pandas as pd
import joblib
import numpy as np

from .recommender import Recommender


class SessionAwareGateRanker(Recommender):
    ANCHOR_WINDOW = 4
    TOPK_PER_SOURCE = 10
    MAX_CANDIDATES = 40

    def __init__(
        self,
        model_path: str,
        recommendations_sasrec_redis,
        recommendations_lfm_redis,
        tracks_redis,
        listen_history_redis,
        baseline_recommender,
        fallback_recommender,
        min_prev_time: float = 0.70,
        abs_threshold: float = 0.72,
        margin: float = 0.08,
    ):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_cols = bundle["feature_cols"]

        self.recommendations_sasrec_redis = recommendations_sasrec_redis
        self.recommendations_lfm_redis = recommendations_lfm_redis
        self.tracks_redis = tracks_redis
        self.listen_history_redis = listen_history_redis

        self.baseline_recommender = baseline_recommender
        self.fallback_recommender = fallback_recommender

        self.min_prev_time = float(min_prev_time)
        self.abs_threshold = float(abs_threshold)
        self.margin = float(margin)

        self._sasrec_cache: Dict[int, List[int]] = {}
        self._lfm_cache: Dict[int, List[int]] = {}
        self._track_cache: Dict[int, object] = {}

    def _safe_baseline(self, user: int, prev_track: int, prev_track_time: float) -> int:
        try:
            rec = self.baseline_recommender.recommend_next(user, prev_track, prev_track_time)
            return int(rec)
        except Exception:
            rec = self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)
            return int(rec)

    def _get_track_info(self, track_id: int):
        if track_id in self._track_cache:
            return self._track_cache[track_id]

        raw = self.tracks_redis.get(track_id)
        if raw is None:
            self._track_cache[track_id] = None
            return None

        try:
            obj = pickle.loads(raw)
        except Exception:
            obj = None

        self._track_cache[track_id] = obj
        return obj

    def _get_i2i_neighbors(self, redis_conn, cache: Dict[int, List[int]], track_id: int) -> List[int]:
        if track_id in cache:
            return cache[track_id]

        raw = redis_conn.get(track_id)
        if raw is None:
            cache[track_id] = []
            return cache[track_id]

        try:
            recs = [int(x) for x in pickle.loads(raw)]
        except Exception:
            recs = []

        cache[track_id] = recs
        return recs

    def _get_user_history(self, user: int) -> List[Tuple[int, float]]:
        key = f"user:{user}:listens"
        raw_history = self.listen_history_redis.lrange(key, 0, -1)

        history = []
        for item in raw_history:
            try:
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                row = json.loads(item)
                history.append((int(row["track"]), float(row["time"])))
            except Exception:
                continue

        # lpush 进 redis 的，一般最新在前，这里反转成时间正序
        history.reverse()
        return history

    def _candidate_set(self, history: List[Tuple[int, float]], seen: set) -> List[int]:
        candidates = []
        added = set()

        anchors = history[-self.ANCHOR_WINDOW:]
        for track_id, _ in reversed(anchors):
            sasrec_cands = self._get_i2i_neighbors(
                self.recommendations_sasrec_redis,
                self._sasrec_cache,
                track_id,
            )[: self.TOPK_PER_SOURCE]

            lfm_cands = self._get_i2i_neighbors(
                self.recommendations_lfm_redis,
                self._lfm_cache,
                track_id,
            )[: self.TOPK_PER_SOURCE]

            for cand in sasrec_cands + lfm_cands:
                if cand in seen:
                    continue
                if cand in added:
                    continue
                added.add(cand)
                candidates.append(int(cand))

                if len(candidates) >= self.MAX_CANDIDATES:
                    return candidates

        return candidates

    def _rank_features_from_source(self, neighbors: List[int], cand: int) -> Tuple[int, float]:
        if cand not in neighbors:
            return 0, 0.0

        rank = neighbors.index(cand) + 1
        rr = 1.0 / rank
        return rank, rr

    def _build_features(
        self,
        history: List[Tuple[int, float]],
        prev_track: int,
        prev_time: float,
        cand: int,
    ) -> Dict[str, float]:
        recent = history[-self.ANCHOR_WINDOW:]
        recent_tracks = [t for t, _ in recent]
        recent_times = [tm for _, tm in recent]

        avg_time = float(np.mean(recent_times)) if recent_times else 0.0
        good_frac = float(np.mean([tm >= 0.7 for tm in recent_times])) if recent_times else 0.0
        skip_frac = float(np.mean([tm < 0.2 for tm in recent_times])) if recent_times else 0.0

        prev_info = self._get_track_info(prev_track)
        cand_info = self._get_track_info(cand)

        same_artist_last = 0.0
        same_title_prefix = 0.0
        cand_artist_repeat = 0.0

        recent_artists = []
        if prev_info is not None and cand_info is not None:
            prev_artist = getattr(prev_info, "artist", None)
            cand_artist = getattr(cand_info, "artist", None)

            same_artist_last = 1.0 if prev_artist == cand_artist and prev_artist is not None else 0.0

            prev_title = str(getattr(prev_info, "title", "") or "").lower()
            cand_title = str(getattr(cand_info, "title", "") or "").lower()
            if prev_title and cand_title:
                same_title_prefix = 1.0 if prev_title[:6] == cand_title[:6] else 0.0

            for t in recent_tracks:
                info = self._get_track_info(t)
                if info is not None:
                    recent_artists.append(getattr(info, "artist", None))

            cand_artist_repeat = float(sum(a == cand_artist for a in recent_artists if a is not None))

        sasrec_rank_sum_inv = 0.0
        lfm_rank_sum_inv = 0.0
        sasrec_hits = 0.0
        lfm_hits = 0.0
        source_agreement = 0.0

        for anchor_track, anchor_time in reversed(recent):
            sas_neighbors = self._get_i2i_neighbors(
                self.recommendations_sasrec_redis, self._sasrec_cache, anchor_track
            )[: self.TOPK_PER_SOURCE]
            lfm_neighbors = self._get_i2i_neighbors(
                self.recommendations_lfm_redis, self._lfm_cache, anchor_track
            )[: self.TOPK_PER_SOURCE]

            sas_rank, sas_rr = self._rank_features_from_source(sas_neighbors, cand)
            lfm_rank, lfm_rr = self._rank_features_from_source(lfm_neighbors, cand)

            if sas_rank > 0:
                sasrec_hits += 1.0
                sasrec_rank_sum_inv += sas_rr * max(anchor_time, 0.05)

            if lfm_rank > 0:
                lfm_hits += 1.0
                lfm_rank_sum_inv += lfm_rr * max(anchor_time, 0.05)

            if sas_rank > 0 and lfm_rank > 0:
                source_agreement += 1.0

        feats = {
            "hist_len": float(len(history)),
            "recent_avg_time": avg_time,
            "recent_last_time": float(prev_time),
            "recent_good_frac": good_frac,
            "recent_skip_frac": skip_frac,
            "same_artist_last": same_artist_last,
            "same_title_prefix": same_title_prefix,
            "cand_artist_repeat": cand_artist_repeat,
            "sasrec_hits": sasrec_hits,
            "lfm_hits": lfm_hits,
            "sasrec_rank_sum_inv": sasrec_rank_sum_inv,
            "lfm_rank_sum_inv": lfm_rank_sum_inv,
            "source_agreement": source_agreement,
        }
        return feats

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        baseline = self._safe_baseline(user, prev_track, prev_track_time)

        try:
            prev_time = float(prev_track_time)
        except Exception:
            return baseline

        # 低质量状态不接管，直接交给 baseline
        if prev_time < self.min_prev_time:
            return baseline

        history = self._get_user_history(user)
        if not history:
            return baseline

        seen = {track for track, _ in history}
        candidates = self._candidate_set(history, seen)

        # 把 baseline 也拉进待比较候选
        if baseline not in candidates and baseline != prev_track:
            candidates = [baseline] + candidates

        # 去掉自己、去重
        cleaned = []
        added = set()
        for cand in candidates:
            cand = int(cand)
            if cand == prev_track:
                continue
            if cand in added:
                continue
            added.add(cand)
            cleaned.append(cand)

        candidates = cleaned
        if not candidates:
            return baseline

        rows = []
        valid_candidates = []

        for cand in candidates:
            feats = self._build_features(history, prev_track, prev_time, cand)
            if feats is None:
                continue

            rows.append(feats)
            valid_candidates.append(cand)

        if not rows:
            return baseline

        X = pd.DataFrame(rows)
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_cols]

        try:
            scores = self.model.predict_proba(X)[:, 1]
        except Exception:
            try:
                scores = self.model.predict(X)
            except Exception:
                return baseline

        score_map = {cand: float(score) for cand, score in zip(valid_candidates, scores)}
        baseline_score = score_map.get(baseline, None)
        if baseline_score is None:
            return baseline

        best_idx = int(np.argmax(scores))
        best_cand = int(valid_candidates[best_idx])
        best_score = float(scores[best_idx])

        # artist 重复惩罚：如果最近已经连续很多同 artist，不轻易替换
        best_info = self._get_track_info(best_cand)
        recent_infos = [self._get_track_info(t) for t, _ in history[-3:]]
        recent_artists = [getattr(x, "artist", None) for x in recent_infos if x is not None]
        best_artist = getattr(best_info, "artist", None) if best_info is not None else None
        same_artist_recent = sum(a == best_artist for a in recent_artists if a is not None)

        should_override = (
            best_cand != baseline
            and best_score >= self.abs_threshold
            and best_score >= baseline_score + self.margin
            and same_artist_recent <= 2
        )

        if should_override:
            return best_cand

        return baseline