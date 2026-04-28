# botify/recommenders/myrec.py

import json
import pickle
import random
from collections import defaultdict

from .recommender import Recommender


MAX_HISTORY = 15


class MyRec(Recommender):

    def __init__(
        self,
        listen_history_redis,
        i2i_redis,
        fallback_recommender,
        popular_tracks=None,
    ):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback_recommender = fallback_recommender
        self.popular_tracks = popular_tracks or []

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        if not history:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        seen_tracks = {track for track, _ in history}

        for anchor in self._rank_anchors(history):
            candidate = self._recommend_from_anchor(anchor, seen_tracks)
            if candidate is not None:
                return candidate

        return self._recommend_popular(seen_tracks, user, prev_track, prev_track_time)

    def _rank_anchors(self, history):

        track_time = defaultdict(float)
        track_best_rank = {}

        for rank, (track, listened_time) in enumerate(history):
            track_time[track] += max(float(listened_time), 1e-3)
            if track not in track_best_rank:
                track_best_rank[track] = rank

        scored = []
        for track, total_time in track_time.items():
            recency = 1.0 / (1.0 + track_best_rank[track])
            score = recency * total_time
            scored.append((score, track))

        scored.sort(reverse=True)
        return [track for _, track in scored]

    def _recommend_from_anchor(self, anchor: int, seen_tracks: set):
        data = self.i2i_redis.get(anchor)
        if data is None:
            return None

        recommendations = pickle.loads(data)
        for track in recommendations:
            candidate = int(track)
            if candidate not in seen_tracks:
                return candidate
        return None

    def _recommend_popular(self, seen_tracks: set, user: int, prev_track: int, prev_track_time: float) -> int:
        """Популярные треки → fallback_recommender."""
        for track in self.popular_tracks:
            if track not in seen_tracks:
                return track
        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

    def _load_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)

        history = []
        for raw in raw_entries[:MAX_HISTORY]:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history
