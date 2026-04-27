import json
import pickle
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .recommender import Recommender


class HSTUContextReranker(Recommender):
    """Two-stage recommender for the HW2 A/B test.

    Stage 1: offline HSTU candidates, already prepared in data/hstu_recommendations.json.
    Stage 2: online reranking using the current short listen history.
    """

    def __init__(
        self,
        recommendations_redis,
        listen_history_redis,
        track_redis,
        catalog,
        fallback_recommender: Recommender,
        artist_repeat_penalty: float = 0.72,
        weak_listen_threshold: float = 0.20,
    ):
        self.recommendations_redis = recommendations_redis
        self.listen_history_redis = listen_history_redis
        self.track_redis = track_redis
        self.catalog = catalog
        self.fallback_recommender = fallback_recommender
        self.artist_repeat_penalty = artist_repeat_penalty
        self.weak_listen_threshold = weak_listen_threshold
        self._artist_cache: Dict[int, Optional[str]] = {}

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        candidates = self._load_user_candidates(user)
        if not candidates:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        history = self._load_user_history(user)
        seen_tracks = {track for track, _ in history}
        artist_counter = Counter(
            artist for artist in (self._artist(track) for track, _ in history) if artist is not None
        )
        prev_artist = self._artist(prev_track)

        best_track = None
        best_score = float("-inf")

        for rank, track in enumerate(candidates[:80]):
            track = int(track)
            if track in seen_tracks:
                continue

            artist = self._artist(track)
            score = self._rank_score(rank)

            if artist is not None:
                score *= self.artist_repeat_penalty ** artist_counter[artist]

                if artist == prev_artist:
                    if prev_track_time <= self.weak_listen_threshold:
                        score *= 0.15
                    else:
                        score *= 0.55

            if score > best_score:
                best_score = score
                best_track = track

        if best_track is not None:
            return int(best_track)

        for track in candidates[80:]:
            track = int(track)
            if track not in seen_tracks:
                return track

        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

    @staticmethod
    def _rank_score(rank: int) -> float:
        return 1.0 / (1.0 + 0.035 * rank)

    def _load_user_candidates(self, user: int) -> List[int]:
        raw = self.recommendations_redis.get(user)
        if raw is None:
            return []
        try:
            return list(self.catalog.from_bytes(raw))
        except Exception:
            try:
                return list(pickle.loads(raw))
            except Exception:
                return []

    def _load_user_history(self, user: int) -> List[Tuple[int, float]]:
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                entry = json.loads(raw)
                history.append((int(entry["track"]), float(entry["time"])))
            except Exception:
                continue
        return history

    def _artist(self, track: int) -> Optional[str]:
        track = int(track)
        if track in self._artist_cache:
            return self._artist_cache[track]

        raw = self.track_redis.get(track)
        if raw is None:
            self._artist_cache[track] = None
            return None

        try:
            artist = self.catalog.from_bytes(raw).artist
        except Exception:
            artist = None
        self._artist_cache[track] = artist
        return artist
