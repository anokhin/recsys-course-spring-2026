import json
import math
import pickle
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .recommender import Recommender


class RankerRecommender(Recommender):
    ANCHOR_WINDOW = 3
    TOPK_PER_ANCHOR = 5
    MAX_CANDIDATES = 30
    MIN_KEEP_HISTORY_FOR_ML = 1

    def __init__(
        self,
        listen_history_redis,
        sasrec_i2i_redis,
        lightfm_i2i_redis,
        model_path: str,
        features_path: str,
        tracks_json_path: str,
        fallback_recommender,
    ):
        self.listen_history_redis = listen_history_redis
        self.sasrec_i2i_redis = sasrec_i2i_redis
        self.lightfm_i2i_redis = lightfm_i2i_redis
        self.fallback = fallback_recommender

        self.model = lgb.Booster(model_file=str(model_path))
        with open(features_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.feature_names = meta["features"]
        self._name_to_idx = {n: i for i, n in enumerate(self.feature_names)}

        self._track_meta = self._load_tracks(tracks_json_path)

        self._sasrec_cache = {}
        self._lightfm_cache = {}

    @staticmethod
    def _load_tracks(path: str) -> dict:
        out = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                raw_year = r.get("year")
                try:
                    year = int(raw_year) if raw_year not in (None, "", 0) else 0
                except (TypeError, ValueError):
                    year = 0
                out[int(r["track"])] = {
                    "artist": r.get("artist"),
                    "genres": set(r.get("genres") or []),
                    "mood": r.get("mood"),
                    "year": year,
                    "artist_fans": float(r.get("artist_fans") or 0.0),
                }
        return out

    def _load_history(self, user: int):
        key = f"user:{user}:listens"
        raw = self.listen_history_redis.lrange(key, 0, -1)
        hist = []
        for r in raw:
            if isinstance(r, bytes):
                r = r.decode("utf-8")
            e = json.loads(r)
            hist.append((int(e["track"]), float(e["time"])))
        return list(reversed(hist))

    def _i2i_neighbours(self, redis_conn, cache: dict, track: int):
        if track in cache:
            return cache[track]
        data = redis_conn.get(track)
        if data is None:
            cache[track] = []
            return cache[track]
        try:
            recs = [int(x) for x in pickle.loads(data)]
        except Exception:
            recs = []
        cache[track] = recs
        return recs

    def _candidate_set(self, history, seen):
        anchors = history[-self.ANCHOR_WINDOW:]
        cands = set()
        for track, _ in anchors:
            for t in self._i2i_neighbours(
                self.sasrec_i2i_redis, self._sasrec_cache, track
            )[: self.TOPK_PER_ANCHOR]:
                if t not in seen:
                    cands.add(t)
            for t in self._i2i_neighbours(
                self.lightfm_i2i_redis, self._lightfm_cache, track
            )[: self.TOPK_PER_ANCHOR]:
                if t not in seen:
                    cands.add(t)
            if len(cands) >= self.MAX_CANDIDATES:
                break
        return list(cands)[: self.MAX_CANDIDATES]

    def _features_for(self, history, candidates):
        times = [t for _, t in history]
        avg_listen_time = float(np.mean(times)) if times else 0.0
        last_listen_time = times[-1] if times else 0.0
        len_hist = len(history)

        history_artists = []
        history_genres_liked = set()
        history_moods_liked = Counter()
        history_years = []
        for trk, tm in history:
            m = self._track_meta.get(trk)
            if m is None:
                continue
            history_artists.append(m["artist"])
            if tm > 0.5:
                history_genres_liked |= m["genres"]
                history_moods_liked[m["mood"]] += 1
            if m["year"] > 0:
                history_years.append(m["year"])
        mean_year = float(np.mean(history_years)) if history_years else 0.0
        num_unique_artists_hist = len(set(history_artists))
        artist_counter = Counter(history_artists)

        anchors = history[-self.ANCHOR_WINDOW:]

        feats = np.zeros((len(candidates), len(self.feature_names)), dtype=np.float64)

        for row_idx, cand in enumerate(candidates):
            cm = self._track_meta.get(cand)
            if cm is None:
                continue

            cand_artist_count = artist_counter.get(cm["artist"], 0)
            cand_artist_discount = 0.8 ** cand_artist_count
            cand_artist_fans_log = math.log1p(cm["artist_fans"])

            if cm["genres"] and history_genres_liked:
                inter = len(cm["genres"] & history_genres_liked)
                uni = len(cm["genres"] | history_genres_liked)
                genre_jaccard = inter / uni if uni else 0.0
            else:
                genre_jaccard = 0.0
            mood_match = history_moods_liked.get(cm["mood"], 0)
            year_dist = (
                abs(cm["year"] - mean_year)
                if cm["year"] > 0 and mean_year > 0
                else 0.0
            )

            sas_hits = lf_hits = 0
            sas_best = lf_best = 11
            sas_w = lf_w = 0.0
            for a_track, a_time in anchors:
                sn = self._i2i_neighbours(
                    self.sasrec_i2i_redis, self._sasrec_cache, a_track
                )
                if cand in sn:
                    rank = sn.index(cand) + 1
                    sas_hits += 1
                    if rank < sas_best:
                        sas_best = rank
                    sas_w += a_time * (11 - rank) / 10.0
                ln = self._i2i_neighbours(
                    self.lightfm_i2i_redis, self._lightfm_cache, a_track
                )
                if cand in ln:
                    rank = ln.index(cand) + 1
                    lf_hits += 1
                    if rank < lf_best:
                        lf_best = rank
                    lf_w += a_time * (11 - rank) / 10.0

            row = {
                "len_hist": len_hist,
                "avg_listen_time": avg_listen_time,
                "last_listen_time": last_listen_time,
                "num_unique_artists_hist": num_unique_artists_hist,
                "cand_artist_count_in_hist": cand_artist_count,
                "cand_artist_discount": cand_artist_discount,
                "cand_artist_fans_log": cand_artist_fans_log,
                "cand_sasrec_hits": sas_hits,
                "cand_sasrec_best_rank": sas_best,
                "cand_sasrec_weighted": sas_w,
                "cand_lightfm_hits": lf_hits,
                "cand_lightfm_best_rank": lf_best,
                "cand_lightfm_weighted": lf_w,
                "cand_genre_jaccard_liked": genre_jaccard,
                "cand_mood_match_count": mood_match,
                "cand_year_abs_distance": year_dist,
            }
            for name, val in row.items():
                idx = self._name_to_idx.get(name)
                if idx is not None:
                    feats[row_idx, idx] = val

        return feats

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        def _safe_fallback():
            r = self.fallback.recommend_next(user, prev_track, prev_track_time)
            return int(r) if r is not None else 0

        history = self._load_history(user)
        seen = {t for t, _ in history}

        if len(history) < self.MIN_KEEP_HISTORY_FOR_ML:
            return _safe_fallback()

        candidates = self._candidate_set(history, seen)
        if not candidates:
            return _safe_fallback()

        valid = [c for c in candidates if c in self._track_meta]
        if not valid:
            return _safe_fallback()

        try:
            feats = self._features_for(history, valid)
            scores = self.model.predict(feats)
            best_idx = int(np.argmax(scores))
            return int(valid[best_idx])
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Ranker failed; falling back")
            return _safe_fallback()