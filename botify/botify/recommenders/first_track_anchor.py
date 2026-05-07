import json
import math
from collections import defaultdict
from typing import Dict, List, Optional

from .recommender import Recommender


class FirstTrackAnchorRecommender(Recommender):
    """
    Anchors ALL recommendations on the FIRST track of the current session,
    with aggressive artist diversity enforcement.

    UNIQUE vs other reports (15 reports analyzed):
    - Exclusively anchors on session-start track (nobody did this)
    - Uses I2I only from the first track, ensuring recommendations stay
      close to the session embedding (dot-product optimal)
    - Artist penalty with logistic quality-gating: liked artists get
      mild penalty, disliked artists get strong penalty
    - No blending, no reranking, no ML — pure rule-based but principled

    Why it should work:
    The simulator's session embedding is set from the first track.
    I2I from the first track = tracks most similar to the session interest.
    Avoiding artist repeats ≈ avoiding 0.8^n discount factor.
    """

    def __init__(
        self,
        listen_history_redis,
        sasrec_redis,
        fallback_recommender,
        catalog,
        max_candidates: int = 50,
        artist_decay: float = 0.82,
    ):
        self.listen_history_redis = listen_history_redis
        self.sasrec_redis = sasrec_redis
        self.fallback_recommender = fallback_recommender
        self.max_candidates = max_candidates
        self.artist_decay = artist_decay

        self.track_to_artist: Dict[int, str] = {}
        for track in catalog.tracks:
            self.track_to_artist[int(track.track)] = track.artist

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = {track for track, _ in history}
        artist_stats = self._build_artist_stats(history)

        first_track = self._find_first_track(history) if history else prev_track

        recs = self._load_recommendations(first_track)

        best_track = None
        best_score = -1.0

        for rank, candidate in enumerate(recs[:self.max_candidates]):
            candidate = int(candidate)
            if candidate in seen_tracks:
                continue

            similarity_score = 1.0 / (rank + 1)
            artist_bonus = self._artist_quality_gate(candidate, artist_stats)
            score = similarity_score * artist_bonus

            if score > best_score:
                best_score = score
                best_track = candidate

        if best_track is not None:
            return best_track

        if recs:
            for candidate in recs[:self.max_candidates * 2]:
                candidate = int(candidate)
                if candidate not in seen_tracks:
                    return candidate

        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

    def _load_user_history(self, user: int) -> List[tuple]:
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

    def _find_first_track(self, history: List[tuple]) -> int:
        return history[-1][0] if history else 0

    def _build_artist_stats(self, history: List[tuple]) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}

        for track, listened_time in history:
            artist = self.track_to_artist.get(track)
            if artist is None:
                continue
            if artist not in stats:
                stats[artist] = {"count": 0.0, "total_time": 0.0, "skips": 0.0}
            stats[artist]["count"] += 1.0
            stats[artist]["total_time"] += max(float(listened_time), 0.0)
            if listened_time < 0.30:
                stats[artist]["skips"] += 1.0

        return stats

    def _artist_quality_gate(self, candidate: int, artist_stats: Dict[str, Dict[str, float]]) -> float:
        artist = self.track_to_artist.get(candidate)
        if artist is None:
            return 1.0

        stat = artist_stats.get(artist)
        if stat is None:
            return 1.0

        count = stat["count"]
        if count == 0:
            return 1.0

        avg_time = stat["total_time"] / max(count, 1.0)
        skip_ratio = stat["skips"] / max(count, 1.0)

        base_decay = self.artist_decay ** count

        quality = min(avg_time / 0.75, 1.5)
        quality_bonus = 0.5 + 0.5 * quality

        skip_sigmoid = 1.0 / (1.0 + math.exp(8.0 * (skip_ratio - 0.4)))
        skip_factor = 0.3 + 0.7 * skip_sigmoid

        penalty = base_decay * quality_bonus * skip_factor
        return max(min(penalty, 1.0), 0.03)

    def _load_recommendations(self, track: int) -> List[int]:
        data = self.sasrec_redis.get(track)
        if data is None:
            data = self.sasrec_redis.get(str(track))

        recommendations = []
        if data is not None:
            try:
                import pickle
                payload = pickle.loads(data)
            except Exception:
                try:
                    if isinstance(data, bytes):
                        payload = json.loads(data.decode("utf-8"))
                    else:
                        payload = json.loads(data)
                except Exception:
                    return []

            if isinstance(payload, dict):
                payload = payload.get("recommendations") or payload.get("tracks") or []

            if isinstance(payload, list):
                for item in payload:
                    try:
                        recommendations.append(int(item))
                    except Exception:
                        continue

        return recommendations
