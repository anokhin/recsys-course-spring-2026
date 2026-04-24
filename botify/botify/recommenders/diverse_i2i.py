import json
import pickle
from collections import defaultdict

from .recommender import Recommender


class DiverseI2IRecommender(Recommender):
    def __init__(self, listen_history_redis, i2i_redis, fallback_recommender, catalog):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback_recommender = fallback_recommender
        self.track_artists = {track.track: track.artist for track in catalog.tracks}

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = {track for track, _ in history}
        seen_artists = {
            self.track_artists[track]
            for track, _ in history
            if track in self.track_artists
        }

        backup = None
        for anchor in self._rank_anchors(history):
            recommendations = self._recommendations(anchor)
            for track in recommendations:
                candidate = int(track)
                if candidate in seen_tracks:
                    continue
                if backup is None:
                    backup = candidate
                if self.track_artists.get(candidate) not in seen_artists:
                    return candidate

        if backup is not None:
            return backup

        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

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

    def _rank_anchors(self, history):
        track_time = defaultdict(float)
        recency = {}

        for index, (track, listened_time) in enumerate(history):
            track_time[track] += listened_time
            recency.setdefault(track, -index)

        anchors = [
            (listened_time, recency[track], track)
            for track, listened_time in track_time.items()
            if listened_time > 0
        ]
        anchors.sort(reverse=True)
        return [track for _, _, track in anchors]

    def _recommendations(self, anchor: int):
        data = self.i2i_redis.get(anchor)
        if data is None:
            return []
        return pickle.loads(data)
