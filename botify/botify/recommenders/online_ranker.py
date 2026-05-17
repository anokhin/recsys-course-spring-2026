import json
import math
import pickle
from collections import Counter

import numpy as np

from .recommender import Recommender


class OnlineRankerRecommender(Recommender):
    def __init__(
        self,
        listen_history_redis,
        sasrec_i2i_redis,
        lightfm_i2i_redis,
        dlrm_i2i_redis,
        model_path,
        tracks_json_path,
        fallback_recommender,
        min_score=0.55,
    ):
        self.listen_history_redis = listen_history_redis
        self.sasrec_i2i_redis = sasrec_i2i_redis
        self.lightfm_i2i_redis = lightfm_i2i_redis
        self.dlrm_i2i_redis = dlrm_i2i_redis
        self.fallback = fallback_recommender
        self.min_score = min_score

        with open(model_path, encoding="utf-8") as f:
            model = json.load(f)
        self.features = model["features"]
        self.name_to_idx = {name: idx for idx, name in enumerate(self.features)}
        self.mean = np.asarray(model["mean"], dtype=np.float64)
        self.scale = np.asarray(model["scale"], dtype=np.float64)
        self.scale[self.scale == 0.0] = 1.0
        self.coef = np.asarray(model["coef"], dtype=np.float64)
        self.intercept = float(model["intercept"])
        params = model.get("params", {})
        self.anchor_window = int(params.get("anchor_window", 4))
        self.per_source = int(params.get("per_source", 10))
        self.max_candidates = int(params.get("max_candidates", 60))
        self.global_stats = {int(k): v for k, v in model.get("global_stats", {}).items()}

        self.track_meta = self._load_tracks(tracks_json_path)
        self.caches = {"sasrec": {}, "lightfm": {}, "dlrm": {}}

    @staticmethod
    def _load_tracks(path):
        meta = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                raw_year = row.get("year")
                try:
                    year = int(raw_year) if raw_year not in (None, "", 0) else 0
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

    def _load_history(self, user):
        raw = self.listen_history_redis.lrange(f"user:{user}:listens", 0, -1)
        history = []
        for item in raw:
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            row = json.loads(item)
            history.append((int(row["track"]), float(row["time"])))
        return list(reversed(history))

    def _neighbors(self, name, redis_conn, track):
        cache = self.caches[name]
        if track in cache:
            return cache[track]
        data = redis_conn.get(track)
        if data is None:
            cache[track] = []
            return cache[track]
        try:
            cache[track] = [int(x) for x in pickle.loads(data)]
        except Exception:
            cache[track] = []
        return cache[track]

    def _candidate_set(self, history, seen):
        cands = []
        added = set()
        for track, _ in reversed(history[-self.anchor_window:]):
            for name, redis_conn in [
                ("sasrec", self.sasrec_i2i_redis),
                ("lightfm", self.lightfm_i2i_redis),
                ("dlrm", self.dlrm_i2i_redis),
            ]:
                for cand in self._neighbors(name, redis_conn, int(track))[: self.per_source]:
                    if cand not in seen and cand not in added:
                        cands.append(cand)
                        added.add(cand)
                    if len(cands) >= self.max_candidates:
                        return cands
        return cands

    def _features_for(self, history, candidates):
        times = [tm for _, tm in history]
        avg_time = float(np.mean(times)) if times else 0.0
        last_time = float(times[-1]) if times else 0.0
        good_frac = float(np.mean([tm >= 0.7 for tm in times])) if times else 0.0

        artists = []
        liked_genres = set()
        liked_moods = Counter()
        years = []
        for track, tm in history:
            meta = self.track_meta.get(int(track))
            if not meta:
                continue
            artists.append(meta["artist"])
            if tm >= 0.5:
                liked_genres |= meta["genres"]
                liked_moods[meta["mood"]] += 1
            if meta["year"] > 0:
                years.append(meta["year"])
        artist_counts = Counter(artists)
        last_artist = self.track_meta.get(int(history[-1][0]), {}).get("artist") if history else None
        mean_year = float(np.mean(years)) if years else 0.0

        rank_tables = []
        for track, tm in history[-self.anchor_window:]:
            rank_tables.append(
                (
                    float(tm),
                    {
                        "sasrec": {t: r + 1 for r, t in enumerate(self._neighbors("sasrec", self.sasrec_i2i_redis, int(track)))},
                        "lightfm": {t: r + 1 for r, t in enumerate(self._neighbors("lightfm", self.lightfm_i2i_redis, int(track)))},
                        "dlrm": {t: r + 1 for r, t in enumerate(self._neighbors("dlrm", self.dlrm_i2i_redis, int(track)))},
                    },
                )
            )

        X = np.zeros((len(candidates), len(self.features)), dtype=np.float64)
        for row_idx, cand in enumerate(candidates):
            meta = self.track_meta.get(int(cand), {})
            cand_genres = meta.get("genres", set())
            if cand_genres and liked_genres:
                genre_jaccard = len(cand_genres & liked_genres) / max(len(cand_genres | liked_genres), 1)
            else:
                genre_jaccard = 0.0
            year = meta.get("year", 0)
            year_dist = abs(year - mean_year) if year > 0 and mean_year > 0 else 0.0
            row = {
                "hist_len": len(history),
                "avg_time": avg_time,
                "last_time": last_time,
                "good_frac": good_frac,
                "unique_artists": len(set(artists)),
                "cand_artist_count": artist_counts.get(meta.get("artist"), 0),
                "same_last_artist": 1.0 if meta.get("artist") == last_artist and last_artist is not None else 0.0,
                "genre_jaccard_good": genre_jaccard,
                "mood_match_count": liked_moods.get(meta.get("mood"), 0),
                "year_abs_distance": year_dist / 50.0,
                "artist_fans_log": math.log1p(float(meta.get("fans", 0.0))),
            }
            source_hits = 0
            for source_name, prefix in [("sasrec", "sasrec"), ("lightfm", "lightfm"), ("dlrm", "dlrm")]:
                hit = 0
                best_inv = 0.0
                weighted = 0.0
                for anchor_time, tables in rank_tables:
                    rank = tables[source_name].get(int(cand))
                    if rank is not None:
                        hit += 1
                        best_inv = max(best_inv, 1.0 / rank)
                        weighted += anchor_time / rank
                row[f"{prefix}_hit"] = hit
                row[f"{prefix}_best_rank_inv"] = best_inv
                row[f"{prefix}_weighted"] = weighted
                source_hits += 1 if hit > 0 else 0
            row["source_hits"] = source_hits

            st = self.global_stats.get(int(cand), [0.0, 0.0, 0.0, 0.0])
            cnt = max(float(st[1]), 1.0)
            row["cand_global_mean_time"] = float(st[0]) / cnt
            row["cand_global_good_rate"] = float(st[2]) / cnt
            row["cand_global_skip_rate"] = float(st[3]) / cnt
            row["cand_global_log_count"] = math.log1p(float(st[1]))

            for name, value in row.items():
                idx = self.name_to_idx.get(name)
                if idx is not None:
                    X[row_idx, idx] = float(value)
        return X

    def _predict_proba(self, X):
        Xs = (X - self.mean) / self.scale
        logits = Xs @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_history(user)
        seen = {track for track, _ in history}
        if not history:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        candidates = self._candidate_set(history, seen)
        candidates = [cand for cand in candidates if cand in self.track_meta]
        if not candidates:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        try:
            scores = self._predict_proba(self._features_for(history, candidates))
            best_idx = int(np.argmax(scores))
            if float(scores[best_idx]) < self.min_score:
                return self.fallback.recommend_next(user, prev_track, prev_track_time)
            return int(candidates[best_idx])
        except Exception:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
