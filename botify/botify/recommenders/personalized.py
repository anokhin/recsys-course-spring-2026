from __future__ import annotations

import json
import math
import pickle
from collections import Counter, defaultdict

import numpy as np

from .recommender import Recommender


class Personalized(Recommender):
    """
    Двух-стадийный рекомендер.

    Retrieval. Берём историю прослушиваний текущей сессии, для каждого
    трека-якоря достаём его SasRec top-10 соседей и агрегируем кандидатов
    с весом по (log-)времени прослушки и обратному рангу в списке соседей.
    Первый трек сессии (time=1.0) совпадает с интересом пользователя
    и поэтому даёт самый сильный сигнал. Уже услышанные треки отбрасываем.

    Rerank. Считаем session-вектор из ALS item-факторов (обучен оффлайн
    на собранных логах симулятора с подмесом user-фактора для холодного
    старта), скорим кандидатов как sv · f_c, добавляем в retrieval-скор.

    Artist diversification. Штрафуем каждого кандидата коэффициентом
    ARTIST_DISCOUNT^count, где count — сколько раз исполнитель уже звучал
    в этой сессии. Это напрямую максимизирует ожидаемое время прослушки:
    симулятор использует такой же мультипликативный штраф при выдаче
    playback time.

    Итог — argmax по переранжированным кандидатам. Fallback в
    SasRec-I2I, если по какой-то причине агрегировать кандидатов
    не удалось (пустая история, нет факторов).
    """

    ARTIST_DISCOUNT = 0.5
    USER_PRIOR_WEIGHT = 0.25
    ALS_WEIGHT = 0.5
    HISTORY_DEPTH = 6

    def __init__(
        self,
        listen_history_redis,
        sasrec_i2i_redis,
        tracks_redis,
        artists_redis,
        catalog,
        factors_path: str,
        fallback_recommender,
    ):
        self.listen_history_redis = listen_history_redis
        self.sasrec_i2i_redis = sasrec_i2i_redis
        self.tracks_redis = tracks_redis
        self.artists_redis = artists_redis
        self.catalog = catalog
        self.fallback_recommender = fallback_recommender

        data = np.load(factors_path)
        self.item_factors: np.ndarray = data["item_factors"].astype(np.float32)
        self.user_factors: np.ndarray = data["user_factors"].astype(np.float32)
        self.n_tracks = self.item_factors.shape[0]
        self.n_users = self.user_factors.shape[0]

        self._artist_cache: dict[str, np.ndarray] = {}

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        if not history:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        seen = {t for t, _ in history}

        agg: dict[int, float] = defaultdict(float)
        for h_rank, (anchor, anchor_time) in enumerate(history):
            weight_h = max(anchor_time, 0.1) / (1.0 + 0.25 * h_rank)
            neighbours = self._i2i_of(anchor)
            if not neighbours:
                continue
            for pos, cand in enumerate(neighbours):
                if cand in seen:
                    continue
                agg[cand] += weight_h / math.log2(pos + 2.0)

        if not agg:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        items = np.fromiter(agg.keys(), dtype=np.int64)
        scores = np.fromiter(agg.values(), dtype=np.float32)

        session_vec = self._session_vector(user, history)
        if session_vec is not None:
            mask = (items >= 0) & (items < self.n_tracks)
            if mask.any():
                als_scores = self.item_factors[items[mask]] @ session_vec
                scores[mask] = scores[mask] + self.ALS_WEIGHT * als_scores

        artist_counts = self._history_artist_counts(history)
        if artist_counts:
            cand_artists = [self._artist_of(int(t)) for t in items]
            multipliers = np.ones(items.shape, dtype=np.float32)
            for i, a in enumerate(cand_artists):
                if a in artist_counts:
                    multipliers[i] = self.ARTIST_DISCOUNT ** artist_counts[a]
            scores = scores * multipliers

        best = int(items[int(np.argmax(scores))])
        return best

    def _i2i_of(self, track: int):
        raw = self.sasrec_i2i_redis.get(track)
        if raw is None:
            return None
        try:
            return [int(t) for t in self.catalog.from_bytes(raw)]
        except Exception:
            try:
                return [int(t) for t in pickle.loads(raw)]
            except Exception:
                return None

    def _session_vector(self, user: int, history):
        tail = history[: self.HISTORY_DEPTH]
        valid_pairs = [(t, tm) for t, tm in tail if 0 <= t < self.n_tracks]
        if not valid_pairs:
            return None
        weights = np.log1p(
            np.array([max(tm, 0.05) for _, tm in valid_pairs], dtype=np.float32) * 8.0
        )
        idx = np.array([t for t, _ in valid_pairs], dtype=np.int64)
        sv = (weights[:, None] * self.item_factors[idx]).sum(axis=0)
        if 0 <= user < self.n_users:
            sv = sv + self.USER_PRIOR_WEIGHT * self.user_factors[user]
        n = float(np.linalg.norm(sv))
        if n < 1e-6:
            return None
        return (sv / n).astype(np.float32)

    def _load_user_history(self, user: int):
        # LPUSH кладёт новейшие треки в начало списка, а наш алгоритм ожидает
        # хронологический порядок (первый трек сессии — самый левый), поэтому
        # возвращаем список в развёрнутом виде.
        raw = self.listen_history_redis.lrange(f"user:{user}:listens", 0, -1)
        out = []
        for r in raw:
            if isinstance(r, bytes):
                r = r.decode("utf-8")
            entry = json.loads(r)
            out.append((int(entry["track"]), float(entry["time"])))
        out.reverse()
        return out

    def _history_artist_counts(self, history) -> Counter:
        counts: Counter[str] = Counter()
        for t, _ in history:
            a = self._artist_of(int(t))
            if a:
                counts[a] += 1
        return counts

    def _artist_of(self, track: int):
        raw = self.tracks_redis.get(track)
        if raw is None:
            return None
        try:
            return self.catalog.from_bytes(raw).artist
        except Exception:
            return None
