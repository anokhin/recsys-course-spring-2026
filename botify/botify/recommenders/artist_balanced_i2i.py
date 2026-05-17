import json
import pickle
from collections import defaultdict

from .recommender import Recommender


class ArtistBalancedI2IRecommender(Recommender):
    """
    I2I recommender that prefers artist diversity without a hard filter.

    The recommender collects candidate proposals from the strongest anchors in
    the current session and chooses the candidate with the best priority tuple.
    The tuple strongly prefers unseen artists, then better anchors, then higher
    rank inside the anchor's I2I list.
    """

    def __init__(self, listen_history_redis, i2i_redis, fallback_recommender, catalog):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback_recommender = fallback_recommender
        self.track_artists = {track.track: track.artist for track in catalog.tracks}

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        if not history:
            return self.fallback_recommender.recommend_next(
                user, prev_track, prev_track_time
            )

        proposals = self._collect_proposals(history)
        if not proposals:
            return self.fallback_recommender.recommend_next(
                user, prev_track, prev_track_time
            )

        best_track, _ = max(proposals.items(), key=lambda item: item[1])
        return best_track

    def _collect_proposals(self, history):
        seen_tracks = {track for track, _ in history}
        seen_artists = {
            self.track_artists.get(track)
            for track, _ in history
            if self.track_artists.get(track) is not None
        }
        anchor_stats = self._anchor_stats(history)

        proposals = {}
        for anchor, stats in anchor_stats.items():
            recommendations = self._recommendations(anchor)
            if not recommendations:
                continue

            for rank, track in enumerate(recommendations):
                candidate = int(track)
                if candidate in seen_tracks:
                    continue

                artist = self.track_artists.get(candidate)
                artist_is_new = int(artist not in seen_artists)
                priority = (
                    artist_is_new,
                    stats["total_time"],
                    stats["last_time"],
                    stats["recency"],
                    -rank,
                )

                current = proposals.get(candidate)
                if current is None or priority > current:
                    proposals[candidate] = priority

        return proposals

    def _anchor_stats(self, history):
        stats = defaultdict(
            lambda: {
                "total_time": 0.0,
                "last_time": 0.0,
                "recency": -1,
            }
        )

        for index, (track, listened_time) in enumerate(history):
            entry = stats[int(track)]
            entry["total_time"] += float(listened_time)
            entry["last_time"] = float(listened_time)
            entry["recency"] = index

        return stats

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

    def _recommendations(self, anchor: int):
        data = self.i2i_redis.get(anchor)
        if data is None:
            return []
        return pickle.loads(data)
