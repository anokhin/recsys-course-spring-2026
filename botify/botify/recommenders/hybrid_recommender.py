import json
import math
import os
import pickle
import random
import zlib
from array import array
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .recommender import Recommender


class HybridRecommender(Recommender):
    """
    ML-рекомендер (онлайн learning-to-rank / contextual bandit).

    В отличие от "эвристик поверх SasRec-I2I", здесь ранжирование кандидатов
    происходит предсказательной моделью (SGDRegressor) по признакам (user context + item).
    Обучение идёт онлайн по факту прослушивания трека, который был рекомендован ранее.
    """

    _LAST_REC_KEY = "ml:last_rec:{user}"
    _LAST_X_KEY = "ml:last_x:{user}"

    def __init__(
        self,
        listen_history_redis,
        lightfm_i2i_redis,
        hstu_redis,
        fallback_recommender: Recommender,
        tracks_meta_path: Optional[str] = None,
        n_candidates: int = 120,
        exploration_eps: float = 0.05,
        hasher_features: int = 2**18,
        random_seed: int = 42,
    ):
        self.listen_history_redis = listen_history_redis
        self.lightfm_i2i_redis = lightfm_i2i_redis
        self.hstu_redis = hstu_redis
        self.fallback_recommender = fallback_recommender
        self.n_candidates = int(n_candidates)
        self.exploration_eps = float(exploration_eps)

        self.rng = random.Random(random_seed)

        if tracks_meta_path is None:
            tracks_meta_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../../data/tracks.json")
            )
        self.tracks_meta_path = tracks_meta_path
        self.tracks_meta: Dict[int, dict] = {}
        self.all_track_ids: List[int] = []
        self.default_track_id = 0
        self._load_tracks_meta()

        self.n_features = int(hasher_features)
        self.lr = 0.05
        self.l2 = 1e-6
        self.bias = 0.0
        self.w = array("f", [0.0]) * self.n_features
        self._is_fitted = False
        self._hash_seed = int(random_seed)

    def _load_tracks_meta(self) -> None:
        meta = {}
        try:
            with open(self.tracks_meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    track_id = rec.get("track")
                    if track_id is None:
                        continue
                    meta[int(track_id)] = {
                        "artist_id": rec.get("artist_id"),
                        "artist_country": rec.get("artist_country"),
                        "artist_genre": rec.get("artist_genre"),
                        "artist_fans": rec.get("artist_fans"),
                        "mood": rec.get("mood"),
                        "year": rec.get("year"),
                        "genres": rec.get("genres") or [],
                    }
        except Exception:
            meta = {}

        self.tracks_meta = meta
        self.all_track_ids = sorted(meta.keys()) if meta else list(range(16200))
        self.default_track_id = self.all_track_ids[0] if self.all_track_ids else 0

    def _load_user_history(self, user: int) -> List[Tuple[int, float]]:
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _safe_fallback(self, user: int, prev_track: int, prev_track_time: float) -> int:
        try:
            rec = self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)
            if rec is not None and isinstance(rec, int) and rec >= 0:
                return rec
        except Exception:
            pass
        return self.default_track_id

    def _year_bucket(self, year) -> str:
        try:
            y = int(year)
        except Exception:
            return "unknown"
        if y < 1970:
            return "lt1970"
        if y < 1980:
            return "1970s"
        if y < 1990:
            return "1980s"
        if y < 2000:
            return "1990s"
        if y < 2010:
            return "2000s"
        if y < 2020:
            return "2010s"
        return "2020s+"

    def _item_features(self, track_id: int) -> Dict[str, float]:
        meta = self.tracks_meta.get(track_id)
        if not meta:
            return {"item:known": 0.0}

        feats: Dict[str, float] = {"item:known": 1.0}
        if meta.get("artist_id") is not None:
            feats[f"item:artist_id={meta['artist_id']}"] = 1.0
        if meta.get("artist_country"):
            feats[f"item:artist_country={meta['artist_country']}"] = 1.0
        if meta.get("artist_genre"):
            feats[f"item:artist_genre={meta['artist_genre']}"] = 1.0
        if meta.get("mood"):
            feats[f"item:mood={meta['mood']}"] = 1.0
        feats[f"item:year_bucket={self._year_bucket(meta.get('year'))}"] = 1.0

        fans = meta.get("artist_fans")
        if fans is not None:
            try:
                fans = float(fans)
                feats["item:artist_fans"] = fans / 100.0
            except Exception:
                pass

        for g in (meta.get("genres") or [])[:5]:
            feats[f"item:genre={g}"] = 1.0

        return feats

    def _context_features(self, history: List[Tuple[int, float]]) -> Dict[str, float]:
        feats: Dict[str, float] = {
            "ctx:has_history": 1.0 if history else 0.0,
            "ctx:history_len": float(len(history)),
        }
        if not history:
            return feats

        last_tracks = [t for t, _ in history[:10]]
        last_times = [tm for _, tm in history[:10]]

        feats["ctx:mean_time_10"] = float(sum(last_times) / max(1, len(last_times)))
        feats["ctx:skips_10"] = float(sum(1 for x in last_times if x < 0.3))
        feats["ctx:long_listens_10"] = float(sum(1 for x in last_times if x > 0.9))

        moods = []
        artist_ids = []
        year_buckets = []
        genres = []
        for tid in last_tracks:
            m = self.tracks_meta.get(tid)
            if not m:
                continue
            if m.get("mood"):
                moods.append(m["mood"])
            if m.get("artist_id") is not None:
                artist_ids.append(m["artist_id"])
            year_buckets.append(self._year_bucket(m.get("year")))
            genres.extend((m.get("genres") or [])[:5])

        for mood, cnt in Counter(moods).most_common(3):
            feats[f"ctx:mood_top={mood}"] = float(cnt)
        for aid, cnt in Counter(artist_ids).most_common(3):
            feats[f"ctx:artist_id_top={aid}"] = float(cnt)
        for yb, cnt in Counter(year_buckets).most_common(2):
            feats[f"ctx:year_bucket_top={yb}"] = float(cnt)
        for g, cnt in Counter(genres).most_common(5):
            feats[f"ctx:genre_top={g}"] = float(cnt)

        return feats

    def _pair_features(self, ctx: Dict[str, float], track_id: int) -> Dict[str, float]:
        feats = dict(ctx)
        feats.update(self._item_features(track_id))

        meta = self.tracks_meta.get(track_id)
        if meta:
            if meta.get("mood"):
                feats[f"cross:ctx_mood_top=={meta['mood']}"] = float(
                    ctx.get(f"ctx:mood_top={meta['mood']}", 0.0) > 0
                )
            if meta.get("artist_id") is not None:
                feats[f"cross:ctx_artist_top=={meta['artist_id']}"] = float(
                    ctx.get(f"ctx:artist_id_top={meta['artist_id']}", 0.0) > 0
                )
            for g in (meta.get("genres") or [])[:5]:
                if ctx.get(f"ctx:genre_top={g}", 0.0) > 0:
                    feats["cross:genre_overlap"] = feats.get("cross:genre_overlap", 0.0) + 1.0
        return feats

    def _hashed(self, feats: Dict[str, float]) -> List[Tuple[int, float]]:
        out: List[Tuple[int, float]] = []
        for k, v in feats.items():
            try:
                val = float(v)
            except Exception:
                continue
            if val == 0.0:
                continue
            key = f"{self._hash_seed}:{k}".encode("utf-8", errors="ignore")
            idx = zlib.crc32(key) % self.n_features
            out.append((idx, val))
        return out

    def _predict(self, feats: Dict[str, float]) -> float:
        s = float(self.bias)
        for idx, val in self._hashed(feats):
            s += float(self.w[idx]) * val
        return s

    def _update_sgd(self, feats: Dict[str, float], y: float) -> None:
        x = self._hashed(feats)
        pred = float(self.bias)
        for idx, val in x:
            pred += float(self.w[idx]) * val
        err = pred - float(y)

        self.bias -= self.lr * err

        for idx, val in x:
            w_old = float(self.w[idx])
            grad = err * val + self.l2 * w_old
            self.w[idx] = float(w_old - self.lr * grad)

    def _candidate_tracks_from_hstu(self, user: int, seen_tracks: set) -> List[int]:
        data = self.hstu_redis.get(user)
        if data is None:
            return []
        try:
            recs = pickle.loads(data)
            out = []
            for r in recs:
                tid = int(r)
                if tid not in seen_tracks:
                    out.append(tid)
                    if len(out) >= self.n_candidates // 2:
                        break
            return out
        except Exception:
            return []

    def _candidate_tracks_from_lightfm_i2i(self, history: List[Tuple[int, float]], seen_tracks: set) -> List[int]:
        if not history:
            return []

        track_time = defaultdict(float)
        for track, listened_time in history:
            track_time[track] += listened_time
        anchors = list(track_time.keys())
        anchor_weights = [track_time[t] for t in anchors]

        out: List[int] = []
        while anchors and len(out) < self.n_candidates:
            anchor = self.rng.choices(anchors, weights=anchor_weights, k=1)[0]
            data = self.lightfm_i2i_redis.get(anchor)
            if data is not None:
                try:
                    recs = pickle.loads(data)
                    for r in recs:
                        tid = int(r)
                        if tid not in seen_tracks and tid not in out:
                            out.append(tid)
                            if len(out) >= self.n_candidates:
                                break
                except Exception:
                    pass

            idx = anchors.index(anchor)
            anchors.pop(idx)
            anchor_weights.pop(idx)

        return out

    def _build_candidates(self, user: int, history: List[Tuple[int, float]]) -> List[int]:
        seen_tracks = set(t for t, _ in history)
        candidates: List[int] = []

        candidates.extend(self._candidate_tracks_from_hstu(user, seen_tracks))
        candidates.extend(self._candidate_tracks_from_lightfm_i2i(history, seen_tracks))

        while len(candidates) < min(self.n_candidates, 200):
            tid = self.rng.choice(self.all_track_ids)
            if tid not in seen_tracks:
                candidates.append(tid)

        uniq = []
        used = set()
        for t in candidates:
            if t not in used and t not in seen_tracks:
                used.add(t)
                uniq.append(int(t))
                if len(uniq) >= self.n_candidates:
                    break
        return uniq

    def _try_learn_from_previous(self, user: int, prev_track: int, prev_track_time: float) -> None:
        try:
            last_rec = self.listen_history_redis.get(self._LAST_REC_KEY.format(user=user))
            last_x = self.listen_history_redis.get(self._LAST_X_KEY.format(user=user))
            if last_rec is None or last_x is None:
                return
            if isinstance(last_rec, bytes):
                last_rec = last_rec.decode("utf-8")
            if int(last_rec) != int(prev_track):
                return

            x_dict = pickle.loads(last_x)
            y = math.log1p(max(0.0, float(prev_track_time)))
            self._update_sgd(x_dict, y)
            self._is_fitted = True
        except Exception:
            return

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        try:
            self._try_learn_from_previous(user, prev_track, prev_track_time)

            history = self._load_user_history(user)
            ctx = self._context_features(history)
            candidates = self._build_candidates(user, history)
            if not candidates:
                return self._safe_fallback(user, prev_track, prev_track_time)

            if self.rng.random() < self.exploration_eps:
                chosen = self.rng.choice(candidates)
                x = self._pair_features(ctx, chosen)
                self.listen_history_redis.set(self._LAST_REC_KEY.format(user=user), str(int(chosen)))
                self.listen_history_redis.set(self._LAST_X_KEY.format(user=user), pickle.dumps(x))
                return int(chosen)

            x_dicts = [self._pair_features(ctx, t) for t in candidates]
            if self._is_fitted:
                preds = [self._predict(x) for x in x_dicts]
            else:
                preds = [0.0 for _ in x_dicts]

            best_idx = max(range(len(candidates)), key=lambda i: preds[i])
            chosen = int(candidates[best_idx])

            x = x_dicts[best_idx]
            self.listen_history_redis.set(self._LAST_REC_KEY.format(user=user), str(chosen))
            self.listen_history_redis.set(self._LAST_X_KEY.format(user=user), pickle.dumps(x))
            return chosen
        except Exception:
            return self._safe_fallback(user, prev_track, prev_track_time)
