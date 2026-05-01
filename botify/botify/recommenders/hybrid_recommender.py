import json
import pickle
import random
from typing import List, Tuple
from .recommender import Recommender


class HybridRecommender(Recommender):
    def __init__(self, listen_history_redis, hstu_redis, i2i_redis, fallback_recommender: Recommender):
        self.listen_history_redis = listen_history_redis
        self.hstu_redis = hstu_redis
        self.i2i_redis = i2i_redis
        self.fallback_recommender = fallback_recommender
        self.lr = 0.05
        self.epsilon = 0.1
        self.n_features = 3

    def _get_weights(self, user: int) -> List[float]:
        data = self.listen_history_redis.get(f"ml_weights:{user}")
        if data:
            return json.loads(data.decode("utf-8"))
        return [0.0] * self.n_features

    def _save_weights(self, user: int, weights: List[float]):
        self.listen_history_redis.set(
            f"ml_weights:{user}", json.dumps(weights))

    def _get_candidates(self, user: int, prev_track: int, seen: set) -> List[Tuple[int, List[float]]]:
        candidates = {}
        hstu_data = self.hstu_redis.get(user)
        if hstu_data:
            try:
                recs = pickle.loads(hstu_data)
            except:
                recs = json.loads(hstu_data)
            if isinstance(recs, list):
                for i, t in enumerate(recs[:20]):
                    try:
                        t_id = int(t)
                        if t_id not in seen:
                            candidates[t_id] = [1.0, 1.0 / (i + 1.0), 0.0]
                    except:
                        continue
        if prev_track > 0:
            i2i_data = self.i2i_redis.get(prev_track)
            if i2i_data:
                try:
                    recs = pickle.loads(i2i_data)
                except:
                    recs = json.loads(i2i_data)
                if isinstance(recs, list):
                    for i, t in enumerate(recs[:20]):
                        try:
                            t_id = int(t)
                            if t_id not in seen:
                                score = 1.0 / (i + 1.0)
                                if t_id in candidates:
                                    candidates[t_id][2] = score
                                else:
                                    candidates[t_id] = [1.0, 0.0, score]
                        except:
                            continue
        return list(candidates.items())

    def _load_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            try:
                entry = json.loads(raw.decode("utf-8"))
                if isinstance(entry, dict):
                    history.append((int(entry["track"]), float(entry["time"])))
            except:
                continue
        return history

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        last_x_data = self.listen_history_redis.get(f"ml_last_x:{user}")
        if last_x_data and prev_track_time > 0:
            try:
                last_x = json.loads(last_x_data.decode("utf-8"))
                weights = self._get_weights(user)
                pred = sum(w * x for w, x in zip(weights, last_x))
                error = prev_track_time - pred
                for i in range(self.n_features):
                    weights[i] += self.lr * error * last_x[i]
                self._save_weights(user, weights)
            except:
                pass
        history = self._load_user_history(user)
        seen_tracks = set(t for t, _ in history)
        seen_tracks.add(prev_track)
        candidates = self._get_candidates(user, prev_track, seen_tracks)
        if not candidates:
            res = self.fallback_recommender.recommend_next(
                user, prev_track, prev_track_time)
            return int(res) if res is not None else 0
        weights = self._get_weights(user)
        if random.random() < self.epsilon:
            chosen_track, chosen_x = random.choice(candidates)
        else:
            best_score = -float('inf')
            chosen_track, chosen_x = candidates[0]
            for track_id, features in candidates:
                score = sum(w * x for w, x in zip(weights, features))
                if score > best_score:
                    best_score = score
                    chosen_track = track_id
                    chosen_x = features
        if chosen_x is not None:
            self.listen_history_redis.set(
                f"ml_last_x:{user}", json.dumps(chosen_x))
        return int(chosen_track)
