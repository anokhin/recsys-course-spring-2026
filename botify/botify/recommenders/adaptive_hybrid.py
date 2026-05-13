import json
import math
import pickle
from collections import defaultdict

from .recommender import Recommender


class AdaptiveHybridRecommender(Recommender):
    SOURCE_SASREC = "sasrec"
    SOURCE_LFM = "lfm"
    SOURCE_HSTU = "hstu"
    SOURCE_TRACK = "track"
    SOURCE_BASELINE = "baseline"

    def __init__(
        self,
        listen_history_redis,
        hstu_redis,
        sasrec_redis,
        lightfm_redis,
        tracks_redis,
        catalog,
        baseline_recommender,
        fallback_recommender,
        max_history=5,
        topk_sasrec=14,
        topk_lightfm=8,
        topk_hstu=18,
        topk_track=8,
    ):
        self.listen_history_redis = listen_history_redis
        self.hstu_redis = hstu_redis
        self.sasrec_redis = sasrec_redis
        self.lightfm_redis = lightfm_redis
        self.catalog = catalog
        self.baseline_recommender = baseline_recommender
        self.fallback_recommender = fallback_recommender

        self.max_history = int(max_history)
        self.topk_sasrec = int(topk_sasrec)
        self.topk_lightfm = int(topk_lightfm)
        self.topk_hstu = int(topk_hstu)
        self.topk_track = int(topk_track)

        self._i2i_cache = {}
        self._hstu_cache = {}

        self.track_by_id = {}

        for track in getattr(catalog, "tracks", []):
            try:
                self.track_by_id[int(track.track)] = track
            except Exception:
                continue

    def _safe_int(self, value, default=None):
        try:
            return int(value)
        except Exception:
            return default

    def _loads_pickle(self, raw, default):
        if raw is None:
            return default

        try:
            return pickle.loads(raw)
        except Exception:
            pass

        try:
            return self.catalog.from_bytes(raw)
        except Exception:
            return default

    def _load_history(self, user):
        key = f"user:{int(user)}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, self.max_history - 1)

        history = []

        for raw in raw_entries:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")

                row = json.loads(raw)
                history.append((int(row["track"]), float(row["time"])))
            except Exception:
                continue

        return history

    def _load_i2i(self, source, track):
        track = int(track)
        cache_key = (source, track)

        if cache_key in self._i2i_cache:
            return self._i2i_cache[cache_key]

        if source == self.SOURCE_SASREC:
            redis_conn = self.sasrec_redis
        else:
            redis_conn = self.lightfm_redis

        raw = redis_conn.get(track)
        recs = self._loads_pickle(raw, [])

        result = []

        for x in recs:
            value = self._safe_int(x)
            if value is not None:
                result.append(value)

        self._i2i_cache[cache_key] = result
        return result

    def _load_hstu(self, user):
        user = int(user)

        if user in self._hstu_cache:
            return self._hstu_cache[user]

        raw = self.hstu_redis.get(user)
        recs = self._loads_pickle(raw, [])

        result = []

        for x in recs:
            value = self._safe_int(x)
            if value is not None:
                result.append(value)

        self._hstu_cache[user] = result
        return result

    def _safe_baseline(self, user, prev_track, prev_track_time):
        try:
            rec = self.baseline_recommender.recommend_next(user, prev_track, prev_track_time)
            return int(rec)
        except Exception:
            pass

        try:
            rec = self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)
            return int(rec)
        except Exception:
            return None

    def _add_candidate(self, table, cand, source, value):
        cand = self._safe_int(cand)

        if cand is None:
            return

        row = table[cand]
        row["score"] += float(value)
        row["sources"][source] += float(value)

    def _track_recommendations(self, track_id):
        track = self.track_by_id.get(int(track_id))

        if track is None:
            return []

        recs = getattr(track, "recommendations", None)

        if not recs:
            return []

        result = []

        for x in recs:
            value = self._safe_int(x)
            if value is not None:
                result.append(value)

        return result

    def _history_signal(self, listened_time, pos):
        listened_time = max(0.0, min(1.0, float(listened_time)))

        # Прямо игнорируем почти скипы.
        # Они не должны тянуть похожие треки.
        if listened_time < 0.30:
            return 0.0

        recency = 0.62 ** pos

        # Нелинейно усиливаем реально хорошие прослушивания.
        quality = listened_time * listened_time

        return recency * (0.20 + quality)

    def _build_candidates(self, user, history, seen, baseline, prev_track_time):
        table = defaultdict(lambda: {"score": 0.0, "sources": defaultdict(float)})

        # Baseline — сильный безопасный кандидат.
        if baseline is not None and baseline not in seen:
            self._add_candidate(table, baseline, self.SOURCE_BASELINE, 0.65)

        # Если последний трек был почти скипом, не экспериментируем.
        if prev_track_time < 0.30:
            return table

        # 1) Главный сигнал: последние хорошо прослушанные треки.
        for pos, pair in enumerate(history[: self.max_history]):
            anchor, listened_time = pair
            signal = self._history_signal(listened_time, pos)

            if signal <= 0.0:
                continue

            # SasRec-I2I — главный session-сигнал.
            sasrec_neighbours = self._load_i2i(self.SOURCE_SASREC, anchor)[
                : self.topk_sasrec
            ]

            for rank, cand in enumerate(sasrec_neighbours, start=1):
                if cand in seen:
                    continue

                value = 1.35 * signal / (rank + 1.35)
                self._add_candidate(table, cand, self.SOURCE_SASREC, value)

            # LightFM — дополнительный источник, слабее SasRec.
            lightfm_neighbours = self._load_i2i(self.SOURCE_LFM, anchor)[
                : self.topk_lightfm
            ]

            for rank, cand in enumerate(lightfm_neighbours, start=1):
                if cand in seen:
                    continue

                value = 0.58 * signal / (rank + 1.85)
                self._add_candidate(table, cand, self.SOURCE_LFM, value)

            # Content/track-level recs из каталога, только если трек реально зашел.
            if listened_time >= 0.55:
                track_recs = self._track_recommendations(anchor)[: self.topk_track]

                for rank, cand in enumerate(track_recs, start=1):
                    if cand in seen:
                        continue

                    value = 0.72 * signal / (rank + 1.60)
                    self._add_candidate(table, cand, self.SOURCE_TRACK, value)

        # 2) HSTU — только как слабый персональный tie-breaker.
        # Раньше HSTU слишком шумел, поэтому вес маленький.
        hstu_recs = self._load_hstu(user)

        hstu_weight = 0.20

        if prev_track_time >= 0.70:
            hstu_weight = 0.28

        for rank, cand in enumerate(hstu_recs[: self.topk_hstu], start=1):
            if cand in seen:
                continue

            value = hstu_weight / math.sqrt(rank + 3.0)
            self._add_candidate(table, cand, self.SOURCE_HSTU, value)

        for bad in list(table.keys()):
            if bad in seen:
                del table[bad]

        return table

    def _choose_source_label(self, row):
        if not row["sources"]:
            return "unknown"

        return max(row["sources"].items(), key=lambda x: x[1])[0]

    def _override_margin(self, prev_track_time):
        # Чем лучше пользователь слушал предыдущий трек,
        # тем охотнее идем в session-рекомендацию.
        if prev_track_time >= 0.78:
            return -0.040

        if prev_track_time >= 0.62:
            return -0.015

        if prev_track_time >= 0.45:
            return 0.010

        if prev_track_time >= 0.30:
            return 0.045

        return 999.0

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        user = int(user)
        prev_track = int(prev_track)
        prev_track_time = float(prev_track_time)

        baseline = self._safe_baseline(user, prev_track, prev_track_time)
        history = self._load_history(user)
        seen = {int(track) for track, _ in history}

        if not history:
            if baseline is not None:
                return baseline

            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        candidates = self._build_candidates(user, history, seen, baseline, prev_track_time)

        if not candidates:
            if baseline is not None:
                return baseline

            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        ranked = []

        for cand, row in candidates.items():
            ranked.append((row["score"], int(cand), row))

        ranked.sort(reverse=True)

        best_score, best, best_row = ranked[0]

        baseline_score = None

        if baseline in candidates:
            baseline_score = candidates[baseline]["score"]

        use_best = baseline is None

        if baseline is not None and best != baseline:
            margin = self._override_margin(prev_track_time)

            if baseline_score is None or best_score >= baseline_score + margin:
                use_best = True

        if use_best:
            return int(best)

        return int(baseline)