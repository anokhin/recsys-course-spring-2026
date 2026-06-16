import json
import pickle
from collections import Counter

from .recommender import Recommender


class MLReranker(Recommender):
    def __init__(
        self,
        listen_history_redis,
        emb_i2i_redis,
        sasrec_i2i_redis,
        track_features,
        fallback_recommender,
    ):
        self.listen_history_redis = listen_history_redis
        self.emb_i2i_redis = emb_i2i_redis
        self.sasrec_i2i_redis = sasrec_i2i_redis
        self.track_features = track_features
        self.fallback = fallback_recommender

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        try:
            return self._recommend(user, prev_track, prev_track_time)
        except Exception:
            try:
                return self.fallback.recommend_next(user, prev_track, prev_track_time)
            except Exception:
                return prev_track

    def _recommend(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen = set(track for track, _ in history)
        artist_counts = Counter()
        for track_id, _ in history:
            feats = self.track_features.get(track_id)
            if feats:
                artist_counts[feats["artist"]] += 1

        candidates = []
        candidate_sources = {}

        for track_id, _ in history[-3:]:
            for source_key, redis_conn in [
                ("emb", self.emb_i2i_redis),
                ("sas", self.sasrec_i2i_redis),
            ]:
                data = redis_conn.get(track_id)
                if data is None:
                    continue
                try:
                    recs = pickle.loads(data)
                except Exception:
                    continue
                for r in recs:
                    rid = int(r)
                    if rid in seen:
                        continue
                    if rid not in candidate_sources:
                        candidate_sources[rid] = set()
                        candidates.append(rid)
                    candidate_sources[rid].add(source_key)

        if not candidates:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        best_track = None
        best_score = -1.0

        for cand in candidates:
            sources = candidate_sources.get(cand, set())
            n_sources = len(sources)

            feats = self.track_features.get(cand)
            if feats is None:
                continue
            artist = feats["artist"]
            artist_cnt = artist_counts.get(artist, 0)

            score = 0.0
            if "emb" in sources:
                score += 3.0
            if "sas" in sources:
                score += 2.0
            score -= artist_cnt * 0.4
            score += n_sources * 1.5

            if score > best_score:
                best_score = score
                best_track = cand

        if best_track is not None:
            return best_track
        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    def _load_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history
