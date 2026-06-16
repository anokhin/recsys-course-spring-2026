import json
import pickle

import joblib
import numpy as np

from .recommender import Recommender


class MLRanker(Recommender):
    PREV_TIME_GATE = 0.80
    CONFIDENCE_GATE = 0.78
    MARGIN_GATE = 0.10

    def __init__(
        self,
        listen_history_redis,
        sasrec_redis,
        tracks_redis,
        model_path: str,
        baseline_recommender: Recommender,
    ):
        self.listen_history_redis = listen_history_redis
        self.sasrec_redis = sasrec_redis
        self.tracks_redis = tracks_redis
        self.baseline_recommender = baseline_recommender

        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        try:
            return self._recommend_with_ml(user, prev_track, prev_track_time)
        except Exception:
            pass
        try:
            return self.baseline_recommender.recommend_next(user, prev_track, prev_track_time)
        except Exception:
            # Если ничего не отработало: return first SasRec candidate or track 0
            candidates = self._get_sasrec_candidates(prev_track)
            return candidates[0] if candidates else 0

    def _recommend_with_ml(self, user, prev_track, prev_track_time):
        history = self._load_history(user)
        seen_tracks = {t for t, _ in history}

        candidates = self._get_sasrec_candidates(prev_track)
        unseen = [c for c in candidates if c not in seen_tracks]

        if not unseen:
            return self.baseline_recommender.recommend_next(user, prev_track, prev_track_time)

        baseline = self.baseline_recommender.recommend_next(user, prev_track, prev_track_time)

        if prev_track_time < self.PREV_TIME_GATE:
            return baseline

        features = self._build_feature_matrix(prev_track, prev_track_time, candidates, unseen)
        scores = self.model.predict_proba(features)[:, 1]

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_track = unseen[best_idx]

        baseline_score = (
            float(scores[unseen.index(baseline)]) if baseline in unseen else 0.0
        )

        if best_score > self.CONFIDENCE_GATE and best_score > baseline_score + self.MARGIN_GATE:
            return best_track
        return baseline

    def _load_history(self, user: int):
        raw_entries = self.listen_history_redis.lrange(f"user:{user}:listens", 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            e = json.loads(raw)
            history.append((int(e["track"]), float(e["time"])))
        return history

    def _get_sasrec_candidates(self, track: int):
        data = self.sasrec_redis.get(track)
        if data is None:
            return []
        return list(pickle.loads(data))[:20]

    def _build_feature_matrix(self, prev_track, prev_track_time, candidates, unseen):
        sasrec_count = len(candidates)
        prev_artist = self._get_artist(prev_track)
        rows = []
        for candidate in unseen:
            try:
                rank = candidates.index(candidate) + 1
                rr = 1.0 / rank
            except ValueError:
                rank, rr = 0, 0.0
            same_artist = int(
                prev_artist is not None and self._get_artist(candidate) == prev_artist
            )
            rows.append([prev_track_time, same_artist, rank, rr, sasrec_count])
        return np.array(rows, dtype=float)

    def _get_artist(self, track_id: int):
        data = self.tracks_redis.get(track_id)
        if data is None:
            return None
        return pickle.loads(data).artist
