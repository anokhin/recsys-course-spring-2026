import json
import pickle
import random
from collections import defaultdict

from .recommender import Recommender


class ContentMMR(Recommender):
    """
    Content-Based recommender with Maximal Marginal Relevance (MMR) re-ranking.

    Strategy:
    1. Load user's listen history (track, time) from Redis.
    2. Collect candidate tracks from the content-based i2i index
       anchored on the most-listened tracks in history.
    3. Re-rank candidates with MMR:
         score = alpha * relevance - (1 - alpha) * max_sim_to_recent
       where relevance = weighted avg cosine sim to history anchors,
       and max_sim_to_recent = penalty for tracks similar to the
       last few tracks (diversity).
    4. Return the highest-scoring unseen track.
    """

    ALPHA = 0.65          # weight: relevance vs. diversity
    RECENT_WINDOW = 5     # how many recent tracks to penalise similarity to
    HISTORY_LIMIT = 10    # tracks to use as anchors
    CANDIDATES_PER_ANCHOR = 20  # neighbors to pull per anchor

    def __init__(
        self,
        listen_history_redis,
        content_i2i_redis,
        track_features: dict,
        fallback_recommender,
    ):
        self.listen_history_redis = listen_history_redis
        self.content_i2i_redis = content_i2i_redis
        self.track_features = track_features   # track_id -> np.ndarray (unit vector)
        self.fallback_recommender = fallback_recommender

    # ------------------------------------------------------------------
    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        if not history:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        seen_tracks = {track for track, _ in history}

        # Weighted anchor tracks (by cumulative listen time)
        track_time: dict = defaultdict(float)
        for track, t in history:
            track_time[track] += t

        # Top anchors by listen time — limited to HISTORY_LIMIT
        anchors = sorted(track_time, key=track_time.__getitem__, reverse=True)[:self.HISTORY_LIMIT]
        anchor_weights = [track_time[a] for a in anchors]
        total_w = sum(anchor_weights) or 1.0
        anchor_weights = [w / total_w for w in anchor_weights]

        # Build candidate set from content-i2i for each anchor
        candidates: set = set()
        for anchor in anchors:
            neighbors = self._get_neighbors(anchor)
            candidates.update(neighbors)
        candidates -= seen_tracks

        if not candidates:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        # Recent tracks for diversity penalty
        recent = [track for track, _ in history[:self.RECENT_WINDOW]]

        # MMR scoring
        best_track = self._mmr_select(
            list(candidates), anchors, anchor_weights, recent
        )
        return best_track if best_track is not None else \
            self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

    # ------------------------------------------------------------------
    def _mmr_select(self, candidates, anchors, anchor_weights, recent):
        """Return the candidate with the highest MMR score."""
        best_score = float("-inf")
        best_track = None

        for candidate in candidates:
            relevance = self._relevance(candidate, anchors, anchor_weights)
            diversity = self._diversity_penalty(candidate, recent)
            score = self.ALPHA * relevance - (1 - self.ALPHA) * diversity
            if score > best_score:
                best_score = score
                best_track = candidate

        return best_track

    def _relevance(self, track_id, anchors, anchor_weights):
        """Weighted average cosine similarity to anchor tracks."""
        vec = self.track_features.get(track_id)
        if vec is None:
            return 0.0
        sim_sum = 0.0
        for anchor, w in zip(anchors, anchor_weights):
            anchor_vec = self.track_features.get(anchor)
            if anchor_vec is not None:
                sim_sum += w * float(vec @ anchor_vec)
        return sim_sum

    def _diversity_penalty(self, track_id, recent):
        """Max cosine similarity to any of the recent tracks."""
        vec = self.track_features.get(track_id)
        if vec is None or not recent:
            return 0.0
        max_sim = 0.0
        for r in recent:
            r_vec = self.track_features.get(r)
            if r_vec is not None:
                sim = float(vec @ r_vec)
                if sim > max_sim:
                    max_sim = sim
        return max_sim

    # ------------------------------------------------------------------
    def _get_neighbors(self, track_id: int):
        data = self.content_i2i_redis.get(track_id)
        if data is None:
            return []
        return pickle.loads(data)

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
