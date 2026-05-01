import json
import math
import os
import pickle
import random
import zlib
from array import array
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Set

from .recommender import Recommender


class HybridRecommender(Recommender):

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
                os.path.join(os.path.dirname(__file__),
                             "../../../data/tracks.json")
            )
        self.tracks_meta_path = tracks_meta_path
        self.tracks_meta: Dict[int, dict] = {}
        self.all_track_ids: List[int] = []
        self.default_track_id = 0
        self._load_tracks_meta()

        self.n_features = int(hasher_features)
        self.lr_init = 0.10
        self.lr_min = 0.0005
        self.l2 = 3e-6
        self.bias = 0.0
        self.w = array("f", [0.0] * self.n_features)
        self.m = array("f", [0.0] * self.n_features)
        self._is_fitted = False
        self._hash_seed = int(random_seed)
        self._update_count = 0
        self._grad_clip = 1.5
        self._pred_clip = 10.0
        self._momentum = 0.9

        self._item_hash_cache: Dict[int, List[Tuple[int, float]]] = {}
        self._item_raw_cache: Dict[int, Dict[str, float]] = {}

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
        self.all_track_ids = sorted(
            meta.keys()) if meta else list(range(16200))
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
            rec = self.fallback_recommender.recommend_next(
                user, prev_track, prev_track_time)
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

    def _get_item_hashes(self, track_id: int) -> List[Tuple[int, float]]:
        if track_id not in self._item_hash_cache:
            raw_feats = self._item_features(track_id)
            self._item_raw_cache[track_id] = raw_feats
            self._item_hash_cache[track_id] = self._hashed(raw_feats)
        return self._item_hash_cache[track_id]

    def _item_features(self, track_id: int) -> Dict[str, float]:
        meta = self.tracks_meta.get(track_id)
        if not meta:
            return {"item:known": 0.0}
        feats = {"item:known": 1.0}
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
                feats["item:artist_fans_log"] = math.log1p(float(fans)) / 12.0
            except Exception:
                pass
        for g in (meta.get("genres") or [])[:5]:
            if g:
                feats[f"item:genre={g}"] = 1.0
        return feats

    def _context_features(self, history: List[Tuple[int, float]]) -> Dict[str, float]:
        feats = {"ctx:has_history": 1.0 if history else 0.0,
                 "ctx:history_len": float(len(history))}
        if not history:
            return feats
        last_tracks = [t for t, _ in history[:10]]
        last_times = [tm for _, tm in history[:10]]
        if last_times:
            m_t = sum(last_times) / max(1, len(last_times))
            feats["ctx:mean_time_10"] = float(m_t)
            feats["ctx:skips_10"] = float(
                sum(1 for x in last_times if x < 0.3))
            feats["ctx:long_listens_10"] = float(
                sum(1 for x in last_times if x > 0.9))
            feats["ctx:std_time_10"] = float(
                (sum((t - m_t)**2 for t in last_times) / len(last_times))**0.5)

        moods, artist_ids, genres = [], [], []
        for tid in last_tracks:
            m = self.tracks_meta.get(tid)
            if not m:
                continue
            if m.get("mood"):
                moods.append(m["mood"])
            if m.get("artist_id") is not None:
                artist_ids.append(m["artist_id"])
            genres.extend((m.get("genres") or [])[:5])

        for mood, cnt in Counter(moods).most_common(3):
            feats[f"ctx:mood_top={mood}"] = float(cnt) / 10.0
        for aid, cnt in Counter(artist_ids).most_common(3):
            feats[f"ctx:artist_id_top={aid}"] = float(cnt) / 10.0
        for g, cnt in Counter(genres).most_common(5):
            feats[f"ctx:genre_top={g}"] = float(cnt) / 10.0
        return feats

    def _get_cross_features(self, ctx: Dict[str, float], track_id: int) -> Dict[str, float]:
        cross_feats = {}
        meta = self.tracks_meta.get(track_id)
        raw_item = self._item_raw_cache.get(
            track_id) or self._item_features(track_id)
        if meta:
            if meta.get("mood") and ctx.get(f"ctx:mood_top={meta['mood']}", 0.0) > 0:
                cross_feats["cross:mood_match"] = 1.0
            if meta.get("artist_id") is not None and ctx.get(f"ctx:artist_id_top={meta['artist_id']}", 0.0) > 0:
                cross_feats["cross:artist_match"] = 1.0
            overlap = sum(1.0 for g in (meta.get("genres") or [])[
                          :5] if g and ctx.get(f"ctx:genre_top={g}", 0.0) > 0)
            if overlap > 0:
                cross_feats["cross:genre_overlap"] = overlap
        if "item:artist_fans_log" in raw_item:
            cross_feats["cross:fans_history"] = raw_item["item:artist_fans_log"] * \
                ctx.get("ctx:history_len", 0)
        return cross_feats

    def _hashed(self, feats: Dict[str, float]) -> List[Tuple[int, float]]:
        out = []
        for k, v in feats.items():
            key = f"{self._hash_seed}:{k}".encode("utf-8", errors="ignore")
            idx = zlib.crc32(key) % self.n_features
            sign = 1 if (zlib.crc32(key + b":sign") & 1) else -1
            out.append((idx, sign * float(v)))
        return out

    def _predict_from_array(self, x: List[Tuple[int, float]]) -> float:
        s = float(self.bias)
        for idx, val in x:
            s += float(self.w[idx]) * val
        return max(-self._pred_clip, min(self._pred_clip, s))

    def _update_sgd_from_array(self, x: List[Tuple[int, float]], y: float) -> None:
        pred = self._predict_from_array(x)
        err = pred - float(y)
        lr = max(self.lr_min, self.lr_init * (0.999 ** self._update_count))
        self._update_count += 1
        self.bias -= lr * err
        for idx, val in x:
            w_old = float(self.w[idx])
            grad = err * val + self.l2 * w_old
            grad = max(-self._grad_clip, min(self._grad_clip, grad))
            m_new = self._momentum * float(self.m[idx]) + grad
            self.m[idx] = float(m_new)
            self.w[idx] = float(
                max(-self._pred_clip, min(self._pred_clip, w_old - lr * m_new)))
        self._is_fitted = True

    def _build_candidates(self, user: int, history: List[Tuple[int, float]]) -> List[int]:
        seen = {t for t, _ in history}
        candidates = []
        data = self.hstu_redis.get(user)
        if data:
            try:
                for r in pickle.loads(data):
                    if int(r) not in seen:
                        candidates.append(int(r))
                        if len(candidates) >= self.n_candidates // 2:
                            break
            except Exception:
                pass

        if history:
            track_t = defaultdict(float)
            for t, tm in history:
                track_t[t] += tm
            anchors = sorted(
                track_t.keys(), key=lambda k: track_t[k], reverse=True)[:5]
            for anchor in anchors:
                d = self.lightfm_i2i_redis.get(anchor)
                if d:
                    try:
                        for r in pickle.loads(d):
                            if int(r) not in seen and int(r) not in candidates:
                                candidates.append(int(r))
                                if len(candidates) >= self.n_candidates:
                                    break
                    except Exception:
                        pass

        while len(candidates) < self.n_candidates:
            tid = self.rng.choice(self.all_track_ids)
            if tid not in seen:
                candidates.append(tid)
        return candidates[:self.n_candidates]

    def _try_learn_from_previous(self, user: int, prev_track: int, prev_track_time: float) -> None:
        try:
            last_rec = self.listen_history_redis.get(
                self._LAST_REC_KEY.format(user=user))
            last_x = self.listen_history_redis.get(
                self._LAST_X_KEY.format(user=user))
            if last_rec and last_x and int(last_rec) == int(prev_track):
                self._update_sgd_from_array(pickle.loads(
                    last_x), math.log1p(max(0.0, float(prev_track_time))))
        except Exception:
            pass

    def _heuristic_score(self, ctx: Dict[str, float], track_id: int) -> float:
        score, meta = 0.0, self.tracks_meta.get(track_id)
        if not meta:
            return 0.0
        for g in (meta.get("genres") or [])[:3]:
            if g and ctx.get(f"ctx:genre_top={g}", 0.0) > 0:
                score += 0.5
        if meta.get("mood") and ctx.get(f"ctx:mood_top={meta['mood']}", 0) > 0:
            score += 0.3
        if meta.get("artist_id") is not None and ctx.get(f"ctx:artist_id_top={meta['artist_id']}", 0) > 0:
            score += 0.4
        return score

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        try:
            self._try_learn_from_previous(user, prev_track, prev_track_time)
            history = self._load_user_history(user)
            ctx = self._context_features(history)
            candidates = self._build_candidates(user, history)
            if not candidates:
                return self._safe_fallback(user, prev_track, prev_track_time)

            h_ctx = self._hashed(ctx)
            eps = max(0.01, self.exploration_eps *
                      (0.9998 ** self._update_count))

            if self.rng.random() < eps:
                chosen = self.rng.choice(candidates)
                x = h_ctx + \
                    self._get_item_hashes(
                        chosen) + self._hashed(self._get_cross_features(ctx, chosen))
            else:
                x_lists = [h_ctx + self._get_item_hashes(t) + self._hashed(
                    self._get_cross_features(ctx, t)) for t in candidates]
                preds = [self._predict_from_array(x) if self._is_fitted else self._heuristic_score(
                    ctx, candidates[i]) for i, x in enumerate(x_lists)]
                idx = max(range(len(candidates)), key=lambda i: preds[i])
                chosen, x = candidates[idx], x_lists[idx]

            self.listen_history_redis.set(
                self._LAST_REC_KEY.format(user=user), str(int(chosen)))
            self.listen_history_redis.set(
                self._LAST_X_KEY.format(user=user), pickle.dumps(x))
            return int(chosen)
        except Exception:
            return self._safe_fallback(user, prev_track, prev_track_time)
