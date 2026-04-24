import json
from pathlib import Path

import numpy as np

from .recommender import Recommender


class SessionSemanticRecommender(Recommender):
    def __init__(
        self,
        listen_history_redis,
        catalog,
        embeddings_path,
        fallback_recommender,
        artist_penalty=0.25,
        min_weight=0.05,
        history_limit=6,
    ):
        self.listen_history_redis = listen_history_redis
        self.catalog = catalog
        self.fallback_recommender = fallback_recommender
        self.artist_penalty = artist_penalty
        self.min_weight = min_weight
        self.history_limit = history_limit

        data = np.load(Path(embeddings_path))
        self.item_vectors = np.ascontiguousarray(data["vectors"].astype(np.float32))
        self.neighbors = np.ascontiguousarray(data["neighbors"].astype(np.int32))

        if len(self.catalog.tracks) != self.item_vectors.shape[0]:
            raise ValueError("Track catalog size does not match semantic embeddings")

        artist_names = [
            track.artist
            for track in sorted(self.catalog.tracks, key=lambda item: item.track)
        ]
        artist_to_id = {
            artist: idx for idx, artist in enumerate(sorted(set(artist_names)))
        }
        self.track_artist_ids = np.array(
            [artist_to_id[artist] for artist in artist_names],
            dtype=np.int32,
        )
        self.n_artists = len(artist_to_id)

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        if not history:
            return self.fallback_recommender.recommend_next(
                user, prev_track, prev_track_time
            )

        recent_history = list(reversed(history[: self.history_limit]))
        history_tracks = np.array(
            [track for track, _ in recent_history],
            dtype=np.int32,
        )
        seen_tracks = set(track for track, _ in history)
        weights = np.array(
            [
                max(float(listened_time), self.min_weight)
                for _, listened_time in recent_history
            ],
            dtype=np.float32,
        )

        profile = (
            self.item_vectors[history_tracks] * weights[:, None]
        ).sum(axis=0) / weights.sum()
        profile /= np.linalg.norm(profile) + 1e-8

        candidate_pool = np.unique(self.neighbors[history_tracks].reshape(-1))
        candidate_pool = np.array(
            [track for track in candidate_pool if track not in seen_tracks],
            dtype=np.int32,
        )

        if candidate_pool.size == 0:
            return self.fallback_recommender.recommend_next(
                user, prev_track, prev_track_time
            )

        scores = self.item_vectors[candidate_pool] @ profile
        artist_counts = np.bincount(
            self.track_artist_ids[history_tracks],
            minlength=self.n_artists,
        )
        scores -= self.artist_penalty * artist_counts[
            self.track_artist_ids[candidate_pool]
        ]

        return int(candidate_pool[int(np.argmax(scores))])

    def _load_user_history(self, user: int):
        raw_entries = self.listen_history_redis.lrange(f"user:{user}:listens", 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history
