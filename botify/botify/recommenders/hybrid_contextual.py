import json
import pickle
from collections import defaultdict

from .recommender import Recommender


class HybridContextualRecommender(Recommender):
    def __init__(
        self,
        listen_history_redis,
        track_redis,
        catalog,
        content_i2i_redis,
        lightfm_i2i_redis,
        sasrec_i2i_redis,
        fallback_recommender,
    ):
        self.listen_history_redis = listen_history_redis
        self.track_redis = track_redis
        self.catalog = catalog
        self.sources = (
            (sasrec_i2i_redis, 2.0),
            (content_i2i_redis, 0.45),
            (lightfm_i2i_redis, 0.30),
        )
        self.fallback_recommender = fallback_recommender

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks, seen_artists, track_time = self._build_context(history)
        prev_anchor = self._safe_int(prev_track)

        if prev_anchor is not None:
            seen_tracks.add(prev_anchor)
            track_data = self._load_track(prev_anchor)
            if track_data is not None:
                seen_artists.add(track_data.artist)

        anchors = self._rank_anchors(prev_anchor, prev_track_time, track_time)
        scores = self._score_candidates(anchors, seen_tracks)

        candidate = self._select_candidate(scores, seen_artists, skip_seen_artists=True)
        if candidate is not None:
            return candidate

        candidate = self._select_candidate(scores, seen_artists, skip_seen_artists=False)
        if candidate is not None:
            return candidate

        return self.fallback_recommender.recommend_next(
            user, prev_track, prev_track_time
        )

    def _load_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)

        history = []
        for raw in raw_entries:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                entry = json.loads(raw)
                history.append((int(entry["track"]), float(entry["time"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        return history

    def _build_context(self, history):
        seen_tracks = set()
        seen_artists = set()
        track_time = defaultdict(float)

        for track, listened_time in history:
            seen_tracks.add(track)
            track_time[track] += listened_time
            track_data = self._load_track(track)
            if track_data is not None:
                seen_artists.add(track_data.artist)

        return seen_tracks, seen_artists, track_time

    def _rank_anchors(self, prev_track, prev_track_time, track_time):
        if prev_track is not None and prev_track not in track_time:
            track_time[prev_track] = max(self._safe_float(prev_track_time), 0.0)

        return sorted(
            ((track, time) for track, time in track_time.items() if time > 0),
            key=lambda item: (-item[1], item[0]),
        )

    def _score_candidates(self, anchors, seen_tracks):
        scores = defaultdict(float)
        for anchor_index, (anchor, accumulated_time) in enumerate(anchors):
            anchor_weight = min(1.0, max(0.05, accumulated_time)) / (
                1 + anchor_index
            )
            for redis, source_weight in self.sources:
                recommendations = self._load_recommendations(redis, anchor)
                for rank, track in enumerate(recommendations):
                    candidate = self._safe_int(track)
                    if candidate is None or candidate in seen_tracks:
                        continue
                    scores[candidate] += source_weight * anchor_weight / (rank + 2)
        return scores

    def _select_candidate(self, scores, seen_artists, skip_seen_artists):
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        for candidate, _ in ranked:
            track_data = self._load_track(candidate)
            if track_data is None:
                continue
            if skip_seen_artists and track_data.artist in seen_artists:
                continue
            return candidate
        return None

    def _load_recommendations(self, redis, track):
        data = redis.get(track)
        if data is None:
            return []
        try:
            recommendations = pickle.loads(data)
        except (pickle.UnpicklingError, EOFError, TypeError):
            return []
        if not isinstance(recommendations, list):
            return []
        return recommendations

    def _load_track(self, track):
        track = self._safe_int(track)
        if track is None:
            return None

        data = self.track_redis.get(track)
        if data is None:
            return None

        try:
            return self.catalog.from_bytes(data)
        except (
            pickle.UnpicklingError,
            EOFError,
            AttributeError,
            TypeError,
            ValueError,
        ):
            return None

    def _safe_int(self, value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _safe_float(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
