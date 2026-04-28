import json
import os
import pickle
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from .recommender import Recommender


class UserHSTUHybridReranker(Recommender):
    """
    Baseline-safe multi-source reranker.

    Fixes compared with the previous version:
    1. The SasRec baseline item is always inserted into the candidate pool.
    2. The baseline item and alternative candidates are scored by the same feature builder.
    3. Debug statistics are printed periodically: model loading, override rate,
       candidate size, score margin, and selected source distribution.
    4. If the joblib model cannot be used, the recommender falls back to a simple
       Reciprocal Rank Fusion style score instead of crashing.
    """

    DEFAULT_FEATURE_NAMES = [
        "prev_track_time",
        "hist_len",
        "recent_avg_time",
        "recent_last_time",
        "recent_good_frac",
        "recent_skip_frac",
        "seen_before",
        "same_as_prev",
        "same_artist_prev",
        "same_artist_recent_count",
        "same_artist_hit",
        "sasrec_hit",
        "hstu_hit",
        "lfm_hit",
        "sasrec_rank_inv",
        "hstu_rank_inv",
        "lfm_rank_inv",
        "source_count",
        "source_agreement",
        "is_baseline",
    ]

    SOURCE_WEIGHTS = {
        "sasrec": 1.00,
        "hstu": 0.70,
        "lfm": 0.75,
        "same_artist": 0.20,
    }

    def __init__(
        self,
        listen_history_redis,
        tracks_redis,
        artists_redis,
        sasrec_redis,
        lfm_redis,
        baseline_recommender: Recommender,
        fallback_recommender: Recommender,
        hstu_redis=None,
        model_path: str = "./reranker_lgb.joblib",
        topk_per_source: int = 20,
        history_limit: int = 10,
        min_prev_time: float = 0.55,
        abs_threshold: float = 0.0,
        margin: float = 0.0001,
        rrf_margin: float = 0.006,
        max_same_artist_recent: int = 3,
        debug_every: int = 1000,
    ):
        self.listen_history_redis = listen_history_redis
        self.tracks_redis = tracks_redis
        self.artists_redis = artists_redis
        self.sasrec_redis = sasrec_redis
        self.hstu_redis = hstu_redis
        self.lfm_redis = lfm_redis
        self.baseline_recommender = baseline_recommender
        self.fallback_recommender = fallback_recommender

        self.model_path = model_path
        self.topk_per_source = int(topk_per_source)
        self.history_limit = int(history_limit)
        self.min_prev_time = float(min_prev_time)
        self.abs_threshold = float(abs_threshold)
        self.margin = float(margin)
        self.rrf_margin = float(rrf_margin)
        self.max_same_artist_recent = int(max_same_artist_recent)
        self.debug_every = int(debug_every)

        self.model_bundle = self._load_model(model_path)
        self.model = self._extract_model(self.model_bundle)
        self.feature_names = self._extract_feature_names(self.model_bundle)

        self.total_calls = 0
        self.scored_calls = 0
        self.model_success_calls = 0
        self.model_error_calls = 0
        self.rrf_calls = 0
        self.override_calls = 0
        self.no_candidate_calls = 0
        self.low_prev_time_blocks = 0
        self.repetitive_artist_blocks = 0
        self.candidate_size_sum = 0
        self.best_score_sum = 0.0
        self.baseline_score_sum = 0.0
        self.margin_sum = 0.0
        self.selected_sources = Counter()

        print(
            "[UserHSTUHybridReranker] init",
            "model_path=", self.model_path,
            "model_loaded=", self.model is not None,
            "features=", len(self.feature_names),
            "topk=", self.topk_per_source,
            "thresholds=", (self.min_prev_time, self.abs_threshold, self.margin),
            flush=True,
        )

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        self.total_calls += 1

        baseline = self._safe_baseline(user, prev_track, prev_track_time)
        if baseline is None:
            self._maybe_print_stats()
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)
        baseline = int(baseline)

        history = self._load_user_history(user)
        seen_tracks = {int(track) for track, _ in history}

        candidates, source_info = self._build_candidates(user, prev_track, history)

        # Critical: score the baseline by exactly the same feature builder.
        candidates.add(baseline)
        source_info[baseline]["baseline_hit"] = 1.0

        candidate_list = [
            int(c) for c in candidates
            if int(c) == baseline or int(c) not in seen_tracks
        ]

        if not candidate_list:
            self.no_candidate_calls += 1
            self._maybe_print_stats()
            return baseline

        feature_dicts = [
            self._feature_dict(
                candidate=c,
                baseline=baseline,
                prev_track=prev_track,
                prev_track_time=prev_track_time,
                history=history,
                source_info=source_info,
            )
            for c in candidate_list
        ]

        scores, score_mode = self._score_candidates(feature_dicts, source_info, candidate_list)
        if scores is None or len(scores) != len(candidate_list):
            self.model_error_calls += 1
            self._maybe_print_stats()
            return baseline

        self.scored_calls += 1
        self.candidate_size_sum += len(candidate_list)
        if score_mode == "model":
            self.model_success_calls += 1
        else:
            self.rrf_calls += 1

        best_idx = int(np.argmax(scores))
        best = int(candidate_list[best_idx])
        best_score = float(scores[best_idx])
        baseline_score = self._score_for_item(candidate_list, scores, baseline)
        score_margin = best_score - baseline_score

        self.best_score_sum += best_score
        self.baseline_score_sum += baseline_score
        self.margin_sum += score_margin

        if self._should_override(
            best=best,
            baseline=baseline,
            best_score=best_score,
            baseline_score=baseline_score,
            score_mode=score_mode,
            prev_track_time=prev_track_time,
            history=history,
        ):
            self.override_calls += 1
            self.selected_sources.update([self._source_signature(best, source_info)])
            self._maybe_print_stats()
            return best

        self._maybe_print_stats()
        return baseline

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _build_candidates(self, user: int, prev_track: int, history: Sequence[Tuple[int, float]]):
        candidates = set()
        source_info: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # SasRec-I2I and LightFM-I2I are item-to-item candidate sources.
        # Their redis keys are track ids, so we query them by recent listened tracks.
        anchors = self._weighted_anchors(history, prev_track)
        for anchor, anchor_weight in anchors:
            self._add_i2i_candidates(
                self.sasrec_redis,
                anchor,
                "sasrec",
                candidates,
                source_info,
                anchor_weight,
            )
            self._add_i2i_candidates(
                self.lfm_redis,
                anchor,
                "lfm",
                candidates,
                source_info,
                anchor_weight,
            )

        # HSTU recommendations in this project are user-level:
        #   {"user": 1, "tracks": [track_id_1, track_id_2, ...]}
        # So we must query HSTU redis by user id, not by prev_track.
        if self.hstu_redis is not None:
            self._add_user_candidates(
                self.hstu_redis,
                user,
                "hstu",
                candidates,
                source_info,
                source_weight=1.0,
            )

        for cand in self._same_artist_candidates(prev_track):
            cand = int(cand)
            candidates.add(cand)
            source_info[cand]["same_artist_hit"] += 1.0
            source_info[cand]["source_count"] += 1.0
            source_info[cand]["same_artist_rank_inv"] += 1.0

        return candidates, source_info

    def _weighted_anchors(self, history: Sequence[Tuple[int, float]], prev_track: int):
        if not history:
            return [(int(prev_track), 1.0)]

        track_weight = defaultdict(float)
        for idx, (track, t) in enumerate(history[: self.history_limit]):
            recency = 1.0 / float(idx + 1)
            track_weight[int(track)] += recency * max(float(t), 0.05)
        track_weight[int(prev_track)] += 1.0
        anchors = sorted(track_weight.items(), key=lambda kv: kv[1], reverse=True)
        return anchors[: min(len(anchors), self.history_limit)]

    def _add_i2i_candidates(self, redis_conn, anchor: int, source_name: str, candidates: set,
                            source_info: Dict[int, Dict[str, float]], anchor_weight: float):
        recs = self._load_recommendations(redis_conn, anchor)
        for rank, cand in enumerate(recs[: self.topk_per_source], start=1):
            cand = int(cand)
            if cand == int(anchor):
                continue
            candidates.add(cand)
            source_info[cand][f"{source_name}_hit"] += 1.0
            source_info[cand][f"{source_name}_rank_inv"] += float(anchor_weight) / float(rank)
            source_info[cand][f"{source_name}_best_rank"] = min(
                float(source_info[cand].get(f"{source_name}_best_rank", 10**9)),
                float(rank),
            )
            source_info[cand]["source_count"] += 1.0

    def _add_user_candidates(self, redis_conn, user: int, source_name: str, candidates: set,
                             source_info: Dict[int, Dict[str, float]], source_weight: float = 1.0):
        """Add candidates from a user-level recommender.

        HSTU recommendations are stored as user -> tracks, while SasRec/LightFM are
        item-to-item. This method keeps HSTU separate so that we do not accidentally
        query user recommendations by prev_track.
        """
        recs = self._load_recommendations(redis_conn, user)
        for rank, cand in enumerate(recs[: self.topk_per_source], start=1):
            cand = int(cand)
            candidates.add(cand)
            source_info[cand][f"{source_name}_hit"] += 1.0
            source_info[cand][f"{source_name}_rank_inv"] += float(source_weight) / float(rank)
            source_info[cand][f"{source_name}_best_rank"] = min(
                float(source_info[cand].get(f"{source_name}_best_rank", 10**9)),
                float(rank),
            )
            source_info[cand]["source_count"] += 1.0

    def _same_artist_candidates(self, prev_track: int) -> List[int]:
        track = self._load_track(prev_track)
        if track is None:
            return []
        artist = getattr(track, "artist", None)
        if not artist:
            return []
        data = self.artists_redis.get(artist)
        if data is None:
            return []
        try:
            tracks = pickle.loads(data)
        except Exception:
            return []
        result = [int(t) for t in tracks if int(t) != int(prev_track)]
        random.shuffle(result)
        return result[: max(5, self.topk_per_source // 2)]

    # ------------------------------------------------------------------
    # Features and scoring
    # ------------------------------------------------------------------

    def _feature_dict(self, candidate: int, baseline: int, prev_track: int, prev_track_time: float,
                      history: Sequence[Tuple[int, float]], source_info: Dict[int, Dict[str, float]]):
        times = [float(t) for _, t in history[: self.history_limit]]
        tracks = [int(t) for t, _ in history[: self.history_limit]]

        recent_avg = float(sum(times) / len(times)) if times else float(prev_track_time)
        recent_last = float(times[0]) if times else float(prev_track_time)
        good_frac = float(sum(t >= 0.75 for t in times) / len(times)) if times else float(prev_track_time >= 0.75)
        skip_frac = float(sum(t <= 0.25 for t in times) / len(times)) if times else float(prev_track_time <= 0.25)

        cand_track = self._load_track(candidate)
        prev = self._load_track(prev_track)
        cand_artist = getattr(cand_track, "artist", None) if cand_track is not None else None
        prev_artist = getattr(prev, "artist", None) if prev is not None else None

        same_artist_prev = float(bool(cand_artist and prev_artist and cand_artist == prev_artist))
        same_artist_recent_count = 0.0
        if cand_artist:
            for tr in tracks:
                tr_obj = self._load_track(tr)
                if tr_obj is not None and getattr(tr_obj, "artist", None) == cand_artist:
                    same_artist_recent_count += 1.0

        info = source_info.get(int(candidate), {})
        sasrec_hit = float(info.get("sasrec_hit", 0.0) > 0.0)
        hstu_hit = float(info.get("hstu_hit", 0.0) > 0.0)
        lfm_hit = float(info.get("lfm_hit", 0.0) > 0.0)
        same_artist_hit = float(info.get("same_artist_hit", 0.0) > 0.0)
        unique_sources = sasrec_hit + hstu_hit + lfm_hit + same_artist_hit

        return {
            "prev_track_time": float(prev_track_time),
            "hist_len": float(len(history)),
            "recent_avg_time": recent_avg,
            "recent_last_time": recent_last,
            "recent_good_frac": good_frac,
            "recent_skip_frac": skip_frac,
            "seen_before": float(int(candidate) in set(tracks)),
            "same_as_prev": float(int(candidate) == int(prev_track)),
            "same_artist_prev": same_artist_prev,
            "same_artist_recent_count": float(same_artist_recent_count),
            "same_artist_hit": same_artist_hit,
            "sasrec_hit": sasrec_hit,
            "hstu_hit": hstu_hit,
            "lfm_hit": lfm_hit,
            "sasrec_rank_inv": float(info.get("sasrec_rank_inv", 0.0)),
            "hstu_rank_inv": float(info.get("hstu_rank_inv", 0.0)),
            "lfm_rank_inv": float(info.get("lfm_rank_inv", 0.0)),
            "same_artist_rank_inv": float(info.get("same_artist_rank_inv", 0.0)),
            "source_count": float(info.get("source_count", 0.0)),
            "unique_source_count": float(unique_sources),
            "source_agreement": float(unique_sources >= 2.0),
            "baseline_hit": float(info.get("baseline_hit", 0.0) > 0.0),
            "is_baseline": float(int(candidate) == int(baseline)),
        }

    def _score_candidates(self, feature_dicts: List[Dict[str, float]], source_info, candidate_list):
        if self.model is not None:
            try:
                X = self._make_model_input(feature_dicts)
                if hasattr(self.model, "predict_proba"):
                    return np.asarray(self.model.predict_proba(X)[:, 1], dtype=float), "model"
                if hasattr(self.model, "decision_function"):
                    z = np.asarray(self.model.decision_function(X), dtype=float)
                    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0))), "model"
                if hasattr(self.model, "predict"):
                    y = np.asarray(self.model.predict(X), dtype=float)
                    return y, "model"
            except Exception as exc:
                self.model_error_calls += 1
                if self.model_error_calls <= 5:
                    print("[UserHSTUHybridReranker] model scoring failed, fallback to RRF:", repr(exc), flush=True)

        return self._rrf_scores(source_info, candidate_list), "rrf"

    def _make_model_input(self, feature_dicts: List[Dict[str, float]]):
        rows = [[float(fd.get(name, 0.0)) for name in self.feature_names] for fd in feature_dicts]
        if pd is not None and getattr(self.model_bundle, "get", None) is not None:
            # Many sklearn/lightgbm pipelines work with numpy, but DataFrame is safer
            # when the training script stored named features.
            return pd.DataFrame(rows, columns=self.feature_names)
        return np.asarray(rows, dtype=float)

    def _rrf_scores(self, source_info, candidate_list):
        scores = []
        for cand in candidate_list:
            info = source_info.get(int(cand), {})
            s = 0.0
            s += self.SOURCE_WEIGHTS["sasrec"] * float(info.get("sasrec_rank_inv", 0.0))
            s += self.SOURCE_WEIGHTS["hstu"] * float(info.get("hstu_rank_inv", 0.0))
            s += self.SOURCE_WEIGHTS["lfm"] * float(info.get("lfm_rank_inv", 0.0))
            s += self.SOURCE_WEIGHTS["same_artist"] * float(info.get("same_artist_rank_inv", 0.0))
            # tiny bonus to avoid replacing a strong baseline unless fusion clearly wins
            if info.get("baseline_hit", 0.0) > 0.0:
                s += 0.01
            scores.append(s)
        return np.asarray(scores, dtype=float)

    def _score_for_item(self, candidate_list, scores, item: int) -> float:
        for cand, score in zip(candidate_list, scores):
            if int(cand) == int(item):
                return float(score)
        return 0.0

    def _should_override(self, best: int, baseline: int, best_score: float, baseline_score: float,
                         score_mode: str, prev_track_time: float, history: Sequence[Tuple[int, float]]) -> bool:
        if int(best) == int(baseline):
            return False
        if prev_track_time < self.min_prev_time:
            self.low_prev_time_blocks += 1
            return False
        if self._too_repetitive_artist(best, history):
            self.repetitive_artist_blocks += 1
            return False

        if score_mode == "model":
            if best_score < self.abs_threshold:
                return False
            if best_score - baseline_score < self.margin:
                return False
            return True

        # RRF scores are not probabilities, so only use a relative margin.
        return (best_score - baseline_score) >= self.rrf_margin

    def _too_repetitive_artist(self, candidate: int, history: Sequence[Tuple[int, float]]) -> bool:
        cand = self._load_track(candidate)
        if cand is None:
            return False
        artist = getattr(cand, "artist", None)
        if not artist:
            return False
        count = 0
        for tr, _ in history[: self.history_limit]:
            obj = self._load_track(tr)
            if obj is not None and getattr(obj, "artist", None) == artist:
                count += 1
        return count >= self.max_same_artist_recent

    def _source_signature(self, candidate: int, source_info) -> str:
        info = source_info.get(int(candidate), {})
        parts = []
        for name in ("sasrec", "hstu", "lfm", "same_artist"):
            if info.get(f"{name}_hit", 0.0) > 0.0:
                parts.append(name)
        return "+".join(parts) if parts else "unknown"

    # ------------------------------------------------------------------
    # Debug stats
    # ------------------------------------------------------------------

    def _maybe_print_stats(self):
        if self.debug_every <= 0 or self.total_calls % self.debug_every != 0:
            return
        scored = max(1, self.scored_calls)
        total = max(1, self.total_calls)
        print(
            "[UserHSTUHybridReranker] stats",
            "total=", self.total_calls,
            "scored=", self.scored_calls,
            "model_success=", self.model_success_calls,
            "model_error=", self.model_error_calls,
            "rrf=", self.rrf_calls,
            "override=", self.override_calls,
            "override_rate=", round(self.override_calls / total, 4),
            "avg_candidates=", round(self.candidate_size_sum / scored, 2),
            "avg_best=", round(self.best_score_sum / scored, 4),
            "avg_base=", round(self.baseline_score_sum / scored, 4),
            "avg_margin=", round(self.margin_sum / scored, 4),
            "low_prev_blocks=", self.low_prev_time_blocks,
            "repetitive_artist_blocks=", self.repetitive_artist_blocks,
            "selected_sources=", dict(self.selected_sources.most_common(8)),
            flush=True,
        )

    # ------------------------------------------------------------------
    # Redis/model helpers
    # ------------------------------------------------------------------

    def _safe_baseline(self, user: int, prev_track: int, prev_track_time: float) -> Optional[int]:
        try:
            rec = self.baseline_recommender.recommend_next(user, prev_track, prev_track_time)
            return int(rec) if rec is not None else None
        except Exception:
            return None

    def _load_model(self, model_path: str):
        print("[UserHSTUHybridReranker] model_path =", model_path, flush=True)
        if not model_path or not os.path.exists(model_path):
            print("[UserHSTUHybridReranker] model file not found; using RRF fallback", flush=True)
            return None
        if joblib is None:
            print("[UserHSTUHybridReranker] joblib not available; using RRF fallback", flush=True)
            return None
        try:
            bundle = joblib.load(model_path)
            print("[UserHSTUHybridReranker] model loaded successfully", flush=True)
            return bundle
        except Exception as exc:
            print("[UserHSTUHybridReranker] failed to load model; using RRF fallback:", repr(exc), flush=True)
            return None

    def _extract_model(self, bundle):
        if bundle is None:
            return None
        if isinstance(bundle, dict):
            return bundle.get("model") or bundle.get("clf") or bundle.get("pipeline") or bundle.get("estimator")
        return bundle

    def _extract_feature_names(self, bundle) -> List[str]:
        if isinstance(bundle, dict):
            for key in ("feature_names", "features", "feature_cols", "columns"):
                value = bundle.get(key)
                if value:
                    return [str(x) for x in value]
        return list(self.DEFAULT_FEATURE_NAMES)

    def _load_user_history(self, user: int) -> List[Tuple[int, float]]:
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, self.history_limit - 1)
        history = []
        for raw in raw_entries:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                entry = json.loads(raw)
                history.append((int(entry["track"]), float(entry["time"])))
            except Exception:
                continue
        return history

    def _load_recommendations(self, redis_conn, key: int) -> List[int]:
        if redis_conn is None:
            return []
        data = redis_conn.get(int(key))
        if data is None:
            return []
        try:
            recs = pickle.loads(data)
            return [int(x) for x in recs]
        except Exception:
            return []

    def _load_track(self, track_id: int):
        data = self.tracks_redis.get(int(track_id))
        if data is None:
            return None
        try:
            return pickle.loads(data)
        except Exception:
            return None
