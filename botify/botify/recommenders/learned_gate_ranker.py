"""LightGBM LambdaRank reranker over a SasRec+LightFM candidate pool with HSTU
features and a confidence-gated override of the SasRec-I2I baseline.

Features must match script/train_ranker.py.
"""

import json
import math
import pickle
from collections import Counter
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np

from .recommender import Recommender


FEATURES = [
    "hist_len",
    "avg_time",
    "last_time",
    "good_frac",
    "skip_frac",
    "unique_artists",
    "same_artist_last",
    "cand_artist_repeat",
    "genre_jaccard_liked",
    "mood_match_count",
    "year_dist",
    "artist_fans_log",
    "sasrec_hits",
    "sasrec_best_rr",
    "sasrec_weighted_rr",
    "lfm_hits",
    "lfm_best_rr",
    "lfm_weighted_rr",
    "source_agreement",
    "hstu_rank_inv",
    "hstu_present",
    "cand_global_mean_time",
    "cand_global_good_rate",
    "cand_global_log_count",
]


class LearnedGateRanker(Recommender):
    ANCHOR_WINDOW = 4
    TOPK_PER_SOURCE = 10
    MAX_CANDIDATES = 50

    def __init__(
        self,
        model_path: str,
        meta_path: str,
        tracks_meta_path: str,
        sasrec_redis,
        lightfm_redis,
        hstu_redis,
        listen_history_redis,
        baseline_recommender: Recommender,
        fallback_recommender: Recommender,
        min_prev_time: float = 0.65,
        margin: float = 0.05,
        same_artist_recent_max: int = 2,
    ):
        self.model = lgb.Booster(model_file=model_path)
        meta = json.loads(open(meta_path).read())
        self.feature_cols = meta["feature_cols"]
        self.global_stats = {int(k): v for k, v in meta.get("global_stats", {}).items()}

        self.sasrec_redis = sasrec_redis
        self.lightfm_redis = lightfm_redis
        self.hstu_redis = hstu_redis
        self.listen_history_redis = listen_history_redis
        self.baseline = baseline_recommender
        self.fallback = fallback_recommender

        self.min_prev_time = float(min_prev_time)
        self.margin = float(margin)
        self.same_artist_recent_max = int(same_artist_recent_max)

        self.tracks_meta = self._load_tracks(tracks_meta_path)
        self._sasrec_cache: Dict[int, List[int]] = {}
        self._lfm_cache: Dict[int, List[int]] = {}
        self._hstu_cache: Dict[int, List[int]] = {}

    @staticmethod
    def _load_tracks(path: str):
        meta = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                try:
                    year = int(row.get("year") or 0)
                except (TypeError, ValueError):
                    year = 0
                meta[int(row["track"])] = {
                    "artist": row.get("artist"),
                    "genres": set(row.get("genres") or []),
                    "mood": row.get("mood"),
                    "year": year,
                    "fans": float(row.get("artist_fans") or 0.0),
                }
        return meta

    def _safe_baseline(self, user, prev_track, prev_track_time):
        try:
            r = self.baseline.recommend_next(user, prev_track, prev_track_time)
            if r is not None:
                return int(r)
        except Exception:
            pass
        try:
            r = self.fallback.recommend_next(user, prev_track, prev_track_time)
            if r is not None:
                return int(r)
        except Exception:
            pass
        return int(prev_track)

    def _i2i(self, redis_conn, cache, track):
        if track in cache:
            return cache[track]
        raw = redis_conn.get(track)
        if raw is None:
            cache[track] = []
            return cache[track]
        try:
            data = pickle.loads(raw)
            cache[track] = [int(x) for x in data]
        except Exception:
            cache[track] = []
        return cache[track]

    def _hstu_topn(self, user):
        if user in self._hstu_cache:
            return self._hstu_cache[user]
        raw = self.hstu_redis.get(user)
        if raw is None:
            self._hstu_cache[user] = []
            return self._hstu_cache[user]
        try:
            data = pickle.loads(raw)
            self._hstu_cache[user] = [int(x) for x in data]
        except Exception:
            self._hstu_cache[user] = []
        return self._hstu_cache[user]

    def _user_history(self, user) -> List[Tuple[int, float]]:
        raw = self.listen_history_redis.lrange(f"user:{user}:listens", 0, -1)
        history = []
        for item in raw:
            try:
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                row = json.loads(item)
                history.append((int(row["track"]), float(row["time"])))
            except Exception:
                continue
        history.reverse()  # lpush stores newest first
        return history

    def _candidate_pool(self, history, seen):
        cands = []
        added = set()
        for track, _ in reversed(history[-self.ANCHOR_WINDOW:]):
            sas = self._i2i(self.sasrec_redis, self._sasrec_cache, int(track))
            lfm = self._i2i(self.lightfm_redis, self._lfm_cache, int(track))
            for cand in (sas[: self.TOPK_PER_SOURCE] + lfm[: self.TOPK_PER_SOURCE]):
                cand = int(cand)
                if cand in seen or cand in added:
                    continue
                added.add(cand)
                cands.append(cand)
                if len(cands) >= self.MAX_CANDIDATES:
                    return cands
        return cands

    def _build_feature_matrix(self, history, prev_track, prev_time, candidates, user):
        times = [tm for _, tm in history]
        avg_time = float(np.mean(times)) if times else 0.0
        last_time = float(prev_time)
        good_frac = float(np.mean([t >= 0.7 for t in times])) if times else 0.0
        skip_frac = float(np.mean([t < 0.2 for t in times])) if times else 0.0

        artists = []
        liked_genres = set()
        liked_moods = Counter()
        years = []
        for tr, tm in history:
            m = self.tracks_meta.get(int(tr))
            if not m:
                continue
            artists.append(m["artist"])
            if tm >= 0.5:
                liked_genres |= m["genres"]
                liked_moods[m["mood"]] += 1
            if m["year"] > 0:
                years.append(m["year"])
        artist_counts = Counter(artists)
        last_artist = self.tracks_meta.get(int(prev_track), {}).get("artist")
        mean_year = float(np.mean(years)) if years else 0.0

        rank_tables = []
        for tr, tm in history[-self.ANCHOR_WINDOW:]:
            sas_n = self._i2i(self.sasrec_redis, self._sasrec_cache, int(tr))
            lfm_n = self._i2i(self.lightfm_redis, self._lfm_cache, int(tr))
            sas = {int(t): r + 1 for r, t in enumerate(sas_n[: self.TOPK_PER_SOURCE])}
            lfm = {int(t): r + 1 for r, t in enumerate(lfm_n[: self.TOPK_PER_SOURCE])}
            rank_tables.append((float(tm), sas, lfm))

        hstu_list = self._hstu_topn(int(user))
        hstu_rank = {int(t): r + 1 for r, t in enumerate(hstu_list)}

        n = len(candidates)
        f = len(self.feature_cols)
        col_idx = {name: i for i, name in enumerate(self.feature_cols)}
        X = np.zeros((n, f), dtype=np.float64)

        for row_idx, cand in enumerate(candidates):
            m = self.tracks_meta.get(int(cand), {})
            cand_artist = m.get("artist")
            cand_genres = m.get("genres", set())
            if cand_genres and liked_genres:
                jacc = len(cand_genres & liked_genres) / max(len(cand_genres | liked_genres), 1)
            else:
                jacc = 0.0
            year = m.get("year", 0)
            year_dist = abs(year - mean_year) / 50.0 if year > 0 and mean_year > 0 else 0.0

            sas_hits = lfm_hits = 0
            sas_best = lfm_best = 0.0
            sas_w = lfm_w = 0.0
            agreement = 0
            for atime, sas, lfm in rank_tables:
                sr = sas.get(int(cand))
                lr = lfm.get(int(cand))
                if sr is not None:
                    sas_hits += 1
                    sas_best = max(sas_best, 1.0 / sr)
                    sas_w += atime / sr
                if lr is not None:
                    lfm_hits += 1
                    lfm_best = max(lfm_best, 1.0 / lr)
                    lfm_w += atime / lr
                if sr is not None and lr is not None:
                    agreement += 1

            hr = hstu_rank.get(int(cand))
            hstu_inv = 1.0 / hr if hr is not None else 0.0
            hstu_pres = 1.0 if hr is not None else 0.0

            st = self.global_stats.get(int(cand), [0.0, 0.0, 0.0])
            cnt = max(st[1], 1.0)
            gmean = st[0] / cnt
            ggood = st[2] / cnt
            glog = math.log1p(st[1])

            row = {
                "hist_len": float(len(history)),
                "avg_time": avg_time,
                "last_time": last_time,
                "good_frac": good_frac,
                "skip_frac": skip_frac,
                "unique_artists": float(len(set(artists))),
                "same_artist_last": 1.0 if cand_artist == last_artist and cand_artist is not None else 0.0,
                "cand_artist_repeat": float(artist_counts.get(cand_artist, 0)),
                "genre_jaccard_liked": jacc,
                "mood_match_count": float(liked_moods.get(m.get("mood"), 0)),
                "year_dist": year_dist,
                "artist_fans_log": math.log1p(float(m.get("fans", 0.0))),
                "sasrec_hits": float(sas_hits),
                "sasrec_best_rr": sas_best,
                "sasrec_weighted_rr": sas_w,
                "lfm_hits": float(lfm_hits),
                "lfm_best_rr": lfm_best,
                "lfm_weighted_rr": lfm_w,
                "source_agreement": float(agreement),
                "hstu_rank_inv": hstu_inv,
                "hstu_present": hstu_pres,
                "cand_global_mean_time": gmean,
                "cand_global_good_rate": ggood,
                "cand_global_log_count": glog,
            }
            for k, v in row.items():
                X[row_idx, col_idx[k]] = float(v)
        return X

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        baseline = self._safe_baseline(user, prev_track, prev_track_time)

        # gate: if user already disengaged, don't risk it
        try:
            prev_t = float(prev_track_time)
        except Exception:
            return baseline
        if prev_t < self.min_prev_time:
            return baseline

        history = self._user_history(user)
        if not history:
            return baseline
        seen = {t for t, _ in history}

        candidates = self._candidate_pool(history, seen)
        # always include the baseline pick so we can compare scores
        if baseline not in candidates and baseline not in seen and baseline != int(prev_track):
            candidates.append(baseline)
        candidates = [c for c in candidates if c != int(prev_track)]
        if not candidates:
            return baseline

        try:
            X = self._build_feature_matrix(history, prev_track, prev_t, candidates, user)
            scores = self.model.predict(X)
        except Exception:
            return baseline

        scores = np.asarray(scores, dtype=np.float64)
        best_idx = int(np.argmax(scores))
        best_cand = int(candidates[best_idx])
        best_score = float(scores[best_idx])

        # gate: only override the baseline when the model strictly prefers a
        # different candidate by a clear margin
        try:
            baseline_idx = candidates.index(baseline)
            baseline_score = float(scores[baseline_idx])
        except ValueError:
            return best_cand

        if best_cand == baseline:
            return baseline

        # anti-monotony: don't override into a cluster of same-artist tracks
        best_artist = self.tracks_meta.get(best_cand, {}).get("artist")
        recent_artists = [
            self.tracks_meta.get(int(t), {}).get("artist")
            for t, _ in history[-3:]
        ]
        same_artist_recent = sum(1 for a in recent_artists if a is not None and a == best_artist)

        if (best_score >= baseline_score + self.margin
                and same_artist_recent <= self.same_artist_recent_max):
            return best_cand
        return baseline
