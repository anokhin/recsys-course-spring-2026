import json
import os
import pickle
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple
from functools import lru_cache

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
    Optimized version: adds caching, batch track loading, pre‑computed common features,
    and numpy‑only model input. No change to feature logic or decision rules.
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

        # Setup LRU caches for Redis lookups
        self._get_track_cached = lru_cache(maxsize=10000)(self._load_track_raw)
        self._get_sasrec_cached = lru_cache(maxsize=5000)(self._load_sasrec_raw)
        self._get_lfm_cached = lru_cache(maxsize=5000)(self._load_lfm_raw)
        self._get_hstu_cached = lru_cache(maxsize=2000)(self._load_hstu_raw)

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

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        self.total_calls += 1

        baseline = self._safe_baseline(user, prev_track, prev_track_time)
        if baseline is None:
            self._maybe_print_stats()
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)
        baseline = int(baseline)

        history = self._load_user_history(user)
        hist_tracks = [t for t, _ in history]
        hist_times = [time for _, time in history]

        # Pre‑compute common stats (once per request)
        common = self._compute_common_stats(history, prev_track_time, hist_tracks, hist_times)

        # Batch load artists for history tracks (used in same_artist_recent_count)
        hist_artist_map = self._batch_get_artists(hist_tracks)

        candidates, source_info = self._build_candidates_cached(user, prev_track, history)

        # Baseline must always be in candidate set
        candidates.add(baseline)
        source_info[baseline]["baseline_hit"] = 1.0

        # Remove already seen tracks (except baseline)
        candidate_list = [
            int(c) for c in candidates
            if int(c) == baseline or int(c) not in common["seen_set"]
        ]

        if not candidate_list:
            self.no_candidate_calls += 1
            self._maybe_print_stats()
            return baseline

        # Batch load artists for all candidates
        cand_artist_map = self._batch_get_artists(candidate_list)

        # Build feature matrix as numpy array
        feature_matrix = []
        for cand in candidate_list:
            cand_artist = cand_artist_map.get(cand)
            f = self._feature_dict_optimized(
                cand, baseline, prev_track, prev_track_time,
                common, hist_artist_map, cand_artist, source_info
            )
            # Order according to self.feature_names
            row = [float(f.get(name, 0.0)) for name in self.feature_names]
            feature_matrix.append(row)
        X = np.asarray(feature_matrix, dtype=np.float32)

        # Score candidates
        scores, score_mode = self._score_candidates(X, source_info, candidate_list)
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
        best = candidate_list[best_idx]
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
    # Candidate generation (with caching)
    # ------------------------------------------------------------------

    def _build_candidates_cached(self, user: int, prev_track: int, history: Sequence[Tuple[int, float]]):
        candidates = set()
        source_info: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        anchors = self._weighted_anchors(history, prev_track)
        for anchor, anchor_weight in anchors:
            self._add_i2i_candidates_cached(
                self._get_sasrec_cached, anchor, "sasrec",
                candidates, source_info, anchor_weight
            )
            self._add_i2i_candidates_cached(
                self._get_lfm_cached, anchor, "lfm",
                candidates, source_info, anchor_weight
            )

        if self.hstu_redis is not None:
            self._add_user_candidates_cached(
                self._get_hstu_cached, user, "hstu",
                candidates, source_info, 1.0
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

    def _add_i2i_candidates_cached(self, cached_getter, anchor: int, source_name: str,
                                   candidates: set, source_info: Dict, anchor_weight: float):
        recs = cached_getter(anchor)  # returns list of ints
        for rank, cand in enumerate(recs[: self.topk_per_source], start=1):
            if cand == anchor:
                continue
            candidates.add(cand)
            source_info[cand][f"{source_name}_hit"] += 1.0
            source_info[cand][f"{source_name}_rank_inv"] += anchor_weight / rank
            best_key = f"{source_name}_best_rank"
            if cand not in source_info or best_key not in source_info[cand] or rank < source_info[cand][best_key]:
                source_info[cand][best_key] = float(rank)
            source_info[cand]["source_count"] += 1.0

    def _add_user_candidates_cached(self, cached_getter, user: int, source_name: str,
                                    candidates: set, source_info: Dict, source_weight: float):
        recs = cached_getter(user)
        for rank, cand in enumerate(recs[: self.topk_per_source], start=1):
            candidates.add(cand)
            source_info[cand][f"{source_name}_hit"] += 1.0
            source_info[cand][f"{source_name}_rank_inv"] += source_weight / rank
            best_key = f"{source_name}_best_rank"
            if cand not in source_info or best_key not in source_info[cand] or rank < source_info[cand][best_key]:
                source_info[cand][best_key] = float(rank)
            source_info[cand]["source_count"] += 1.0

    def _same_artist_candidates(self, prev_track: int) -> List[int]:
        track = self._get_track_cached(prev_track)
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
    # Feature computation (optimized)
    # ------------------------------------------------------------------

    def _compute_common_stats(self, history, prev_track_time, hist_tracks, hist_times):
        if hist_times:
            recent_avg = sum(hist_times) / len(hist_times)
            recent_last = hist_times[0]
            good_frac = sum(1 for t in hist_times if t >= 0.75) / len(hist_times)
            skip_frac = sum(1 for t in hist_times if t <= 0.25) / len(hist_times)
        else:
            recent_avg = prev_track_time
            recent_last = prev_track_time
            good_frac = 1.0 if prev_track_time >= 0.75 else 0.0
            skip_frac = 1.0 if prev_track_time <= 0.25 else 0.0

        return {
            "hist_len": len(history),
            "recent_avg_time": recent_avg,
            "recent_last_time": recent_last,
            "recent_good_frac": good_frac,
            "recent_skip_frac": skip_frac,
            "seen_set": set(hist_tracks),
            "prev_track_time": prev_track_time,
            "prev_track": history[0][0] if history else None,
        }

    def _batch_get_artists(self, track_ids: List[int]) -> Dict[int, Optional[str]]:
        if not track_ids:
            return {}
        # Use mget for batch retrieval
        keys = [str(tid) for tid in track_ids]
        raw_data = self.tracks_redis.mget(*keys)
        result = {}
        for tid, raw in zip(track_ids, raw_data):
            if raw:
                try:
                    obj = pickle.loads(raw)
                    result[tid] = getattr(obj, "artist", None)
                except Exception:
                    result[tid] = None
            else:
                result[tid] = None
        return result

    def _feature_dict_optimized(
        self,
        cand: int,
        baseline: int,
        prev_track: int,
        prev_track_time: float,
        common: Dict,
        hist_artist_map: Dict[int, Optional[str]],
        cand_artist: Optional[str],
        source_info: Dict[int, Dict[str, float]],
    ) -> Dict[str, float]:
        info = source_info.get(cand, {})
        prev_artist = hist_artist_map.get(prev_track) if common["prev_track"] is not None else None

        same_artist_prev = 1.0 if (cand_artist and prev_artist and cand_artist == prev_artist) else 0.0
        # Count how many of the recent history tracks share this artist
        same_artist_recent = 0.0
        if cand_artist:
            for a in hist_artist_map.values():
                if a == cand_artist:
                    same_artist_recent += 1.0
        same_artist_recent = same_artist_recent / 10.0  # normalize

        sasrec_hit = 1.0 if info.get("sasrec_hit", 0.0) > 0 else 0.0
        hstu_hit = 1.0 if info.get("hstu_hit", 0.0) > 0 else 0.0
        lfm_hit = 1.0 if info.get("lfm_hit", 0.0) > 0 else 0.0
        same_artist_hit = 1.0 if info.get("same_artist_hit", 0.0) > 0 else 0.0
        unique_sources = sasrec_hit + hstu_hit + lfm_hit + same_artist_hit

        return {
            "prev_track_time": common["prev_track_time"],
            "hist_len": float(common["hist_len"]),
            "recent_avg_time": float(common["recent_avg_time"]),
            "recent_last_time": float(common["recent_last_time"]),
            "recent_good_frac": float(common["recent_good_frac"]),
            "recent_skip_frac": float(common["recent_skip_frac"]),
            "seen_before": 1.0 if cand in common["seen_set"] else 0.0,
            "same_as_prev": 1.0 if cand == prev_track else 0.0,
            "same_artist_prev": same_artist_prev,
            "same_artist_recent_count": same_artist_recent,
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
            "source_agreement": 1.0 if unique_sources >= 2.0 else 0.0,
            "baseline_hit": float(info.get("baseline_hit", 0.0) > 0.0),
            "is_baseline": 1.0 if cand == baseline else 0.0,
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_candidates(self, X: np.ndarray, source_info, candidate_list):
        if self.model is not None:
            try:
                if hasattr(self.model, "predict_proba"):
                    scores = self.model.predict_proba(X)[:, 1]
                elif hasattr(self.model, "decision_function"):
                    z = self.model.decision_function(X)
                    scores = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                elif hasattr(self.model, "predict"):
                    scores = self.model.predict(X).astype(float)
                else:
                    raise RuntimeError("Model has no scoring method")
                return scores, "model"
            except Exception as exc:
                self.model_error_calls += 1
                if self.model_error_calls <= 5:
                    print("[UserHSTUHybridReranker] model scoring failed, fallback to RRF:", repr(exc), flush=True)

        return self._rrf_scores(source_info, candidate_list), "rrf"

    def _rrf_scores(self, source_info, candidate_list):
        scores = np.zeros(len(candidate_list), dtype=float)
        for i, cand in enumerate(candidate_list):
            info = source_info.get(cand, {})
            s = 0.0
            s += self.SOURCE_WEIGHTS["sasrec"] * info.get("sasrec_rank_inv", 0.0)
            s += self.SOURCE_WEIGHTS["hstu"] * info.get("hstu_rank_inv", 0.0)
            s += self.SOURCE_WEIGHTS["lfm"] * info.get("lfm_rank_inv", 0.0)
            s += self.SOURCE_WEIGHTS["same_artist"] * info.get("same_artist_rank_inv", 0.0)
            if info.get("baseline_hit", 0.0) > 0.0:
                s += 0.01
            scores[i] = s
        return scores

    def _score_for_item(self, candidate_list, scores, item: int) -> float:
        for cand, score in zip(candidate_list, scores):
            if int(cand) == int(item):
                return float(score)
        return 0.0

    def _should_override(self, best: int, baseline: int, best_score: float, baseline_score: float,
                         score_mode: str, prev_track_time: float, history: Sequence[Tuple[int, float]]) -> bool:
        if best == baseline:
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
        cand_track = self._get_track_cached(candidate)
        if cand_track is None:
            return False
        artist = getattr(cand_track, "artist", None)
        if not artist:
            return False
        count = 0
        for tr, _ in history[: self.history_limit]:
            tr_track = self._get_track_cached(tr)
            if tr_track is not None and getattr(tr_track, "artist", None) == artist:
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
    # Redis / caching helpers
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

    # Raw cacheable methods (these return deserialized objects)
    def _load_track_raw(self, track_id: int):
        data = self.tracks_redis.get(track_id)
        if data is None:
            return None
        try:
            return pickle.loads(data)
        except Exception:
            return None

    def _load_sasrec_raw(self, anchor: int):
        data = self.sasrec_redis.get(anchor)
        if data is None:
            return []
        try:
            return pickle.loads(data)
        except Exception:
            return []

    def _load_lfm_raw(self, anchor: int):
        data = self.lfm_redis.get(anchor)
        if data is None:
            return []
        try:
            return pickle.loads(data)
        except Exception:
            return []

    def _load_hstu_raw(self, user: int):
        if self.hstu_redis is None:
            return []
        data = self.hstu_redis.get(user)
        if data is None:
            return []
        try:
            return pickle.loads(data)
        except Exception:
            return []