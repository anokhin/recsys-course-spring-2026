import json
import pickle
from collections import defaultdict

from .recommender import Recommender


class LightFMContextualRecommender(Recommender):
    def __init__(self, lfm_redis, listen_history_redis, catalog, fallback):
        self.lfm_redis = lfm_redis
        self.listen_history_redis = listen_history_redis
        self.fallback = fallback
        self.track_artist: dict = {t.track: t.artist for t in catalog.tracks}

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_history(user)
        if not history:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        seen = {t for t, _ in history}
        recent_artists = {
            self.track_artist[t] for t, _ in history if t in self.track_artist
        }

        track_time: dict = defaultdict(float)
        for track, t in history:
            track_time[track] += t
        anchors = sorted(track_time, key=track_time.__getitem__, reverse=True)

        for anchor in anchors:
            candidate = self._first_candidate(anchor, seen, artist_blacklist=recent_artists)
            if candidate is not None:
                return candidate

        for anchor in anchors:
            candidate = self._first_candidate(anchor, seen, artist_blacklist=set())
            if candidate is not None:
                return candidate

        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    def _first_candidate(self, anchor: int, seen: set, artist_blacklist: set):
        raw = self.lfm_redis.get(anchor)
        if raw is None:
            return None
        for t in pickle.loads(raw):
            t = int(t)
            if t in seen:
                continue
            if self.track_artist.get(t) in artist_blacklist:
                continue
            return t
        return None

    def _load_history(self, user: int):
        key = f"user:{user}:listens"
        history = []
        for raw in self.listen_history_redis.lrange(key, 0, -1):
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history
