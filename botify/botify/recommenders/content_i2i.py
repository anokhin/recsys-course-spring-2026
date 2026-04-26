import json
import pickle
from collections import defaultdict

from .recommender import Recommender


class ContentI2IRecommender(Recommender):
    ARTIST_WINDOW = 3

    def __init__(
        self, listen_history_redis, i2i_redis, tracks_redis, catalog, fallback
    ):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.tracks_redis = tracks_redis
        self.catalog = catalog
        self.fallback = fallback

    def recommend_next(
        self, user: int, prev_track: int, prev_track_time: float
    ) -> int:
        history = self._load_history(user)
        seen_tracks = {track for track, _, _ in history}
        recent_artists = self._recent_artists(history)

        if history:
            track_time = defaultdict(float)
            for track, t, _ in history:
                track_time[track] += t

            anchors = sorted(track_time.keys(), key=lambda t: -track_time[t])

            for anchor in anchors:
                candidate = self._pick(anchor, seen_tracks, recent_artists)
                if candidate is not None:
                    return candidate

            for anchor in anchors:
                candidate = self._pick(anchor, seen_tracks, set())
                if candidate is not None:
                    return candidate

        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    def _pick(self, anchor: int, seen_tracks: set, avoid_artists: set):
        data = self.i2i_redis.get(anchor)
        if data is None:
            return None
        recommendations = pickle.loads(data)
        for track in recommendations:
            tid = int(track)
            if tid in seen_tracks:
                continue
            if avoid_artists:
                artist = self._get_artist(tid)
                if artist and artist in avoid_artists:
                    continue
            return tid
        return None

    def _get_artist(self, track_id: int) -> str:
        raw = self.tracks_redis.get(track_id)
        if raw is None:
            return ""
        return self.catalog.from_bytes(raw).artist or ""

    def _recent_artists(self, history) -> set:
        """Last ARTIST_WINDOW unique artists from session history."""
        seen = []
        unique = set()
        for _, _, artist in history:
            if artist and artist not in unique:
                unique.add(artist)
                seen.append(artist)
                if len(seen) >= self.ARTIST_WINDOW:
                    break
        return unique

    def _load_history(self, user: int):
        """Returns list of (track_id, time, artist) most-recent-first."""
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append(
                (
                    int(entry["track"]),
                    float(entry["time"]),
                    entry.get("artist", ""),
                )
            )
        return history
