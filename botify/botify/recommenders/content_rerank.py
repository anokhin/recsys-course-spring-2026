"""Content-aware re-ranker on top of SasRec-I2I.

The recommender keeps a per-track content embedding in memory (built offline
from track metadata: artist, genres, mood, summary, etc.). At every request:

1. Fetch the user's recent listen history from Redis (track id + listen time
   tuples; same key the existing :class:`I2IRecommender` reads from).
2. Estimate the **session intent vector** as a listen-time-weighted average
   of the embeddings of the tracks the user actually listened to in this
   session (skips with very low time are dropped — they pollute the signal).
3. Build a candidate pool by unioning the top-K SasRec-I2I neighbours of every
   anchor in the history (weighted by listen time). Tracks already heard in
   the session are filtered out.
4. Score every candidate by ``cosine(emb, session_vec)`` minus a small
   diversity penalty proportional to how often the candidate's artist already
   appeared in the session — the simulator penalises artist repetition.
5. Return the highest-scoring candidate. Falls back to the supplied fallback
   recommender (SasRec-I2I behaviour) when the history is empty or no
   candidate survives filtering.
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from .recommender import Recommender


class ContentRerankRecommender(Recommender):

    DIVERSITY_PENALTY = 0.07
    MIN_LISTEN_TIME = 0.10
    CANDIDATES_PER_ANCHOR = 10
    MAX_HISTORY_LEN = 50
    SKIP_PENALTY_THRESHOLD = 0.05  # listen time below this: treat as a skip and weight negatively

    def __init__(
        self,
        listen_history_redis,
        i2i_redis,
        embeddings_path: str,
        meta_path: str,
        fallback_recommender,
    ):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.fallback_recommender = fallback_recommender

        self.embeddings = np.load(embeddings_path).astype(np.float32)
        with open(meta_path) as fh:
            meta = json.load(fh)
        self.dim = int(meta["dim"])
        artist_by_track = meta.get("artist_by_track", {})
        # Materialise as an array for fast lookup
        n = self.embeddings.shape[0]
        self.artist_by_track: list[str] = [""] * n
        for tid_str, artist in artist_by_track.items():
            tid = int(tid_str)
            if 0 <= tid < n:
                self.artist_by_track[tid] = artist or ""

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        if not history:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        seen_tracks = {track for track, _ in history}

        # 1. Session intent vector
        session_vec = self._session_vector(history)
        if session_vec is None:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        # 2. Candidate pool: top-K SasRec neighbours per anchor (weighted by listen time)
        candidates = self._collect_candidates(history, seen_tracks)
        if not candidates:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        # 3. Score and pick best
        artist_counts: Counter[str] = Counter()
        for track, time in history:
            if 0 <= track < len(self.artist_by_track):
                artist_counts[self.artist_by_track[track]] += 1

        candidate_ids = np.fromiter(candidates, dtype=np.int64, count=len(candidates))
        embs = self.embeddings[candidate_ids]
        cos_scores = embs @ session_vec
        artist_pen = np.fromiter(
            (
                artist_counts.get(self.artist_by_track[c] if 0 <= c < len(self.artist_by_track) else "", 0)
                for c in candidate_ids
            ),
            dtype=np.float32,
            count=len(candidate_ids),
        )
        scores = cos_scores - self.DIVERSITY_PENALTY * artist_pen

        best_idx = int(np.argmax(scores))
        best_track = int(candidate_ids[best_idx])
        if best_track in seen_tracks:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)
        return best_track

    # ------------------------------------------------------------------ helpers

    def _load_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, self.MAX_HISTORY_LEN - 1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _session_vector(self, history):
        n = self.embeddings.shape[0]
        acc = np.zeros(self.dim, dtype=np.float32)
        total_w = 0.0
        for track, time in history:
            if not (0 <= track < n):
                continue
            if time >= self.MIN_LISTEN_TIME:
                w = float(time)
                acc += self.embeddings[track] * w
                total_w += w
            elif time < self.SKIP_PENALTY_THRESHOLD:
                # Subtract a small amount for instant skips
                acc -= self.embeddings[track] * 0.2
        if total_w <= 0 and not np.any(acc):
            return None
        norm = float(np.linalg.norm(acc))
        if norm <= 1e-8:
            return None
        return (acc / norm).astype(np.float32)

    def _collect_candidates(self, history, seen_tracks):
        """Union of SasRec-I2I top-K for each anchor track, ignoring already-heard."""
        candidates: set[int] = set()
        # Anchor weighting by listen time encourages neighbours of well-liked tracks.
        ordered = sorted(history, key=lambda x: x[1], reverse=True)
        for track, _ in ordered:
            data = self.i2i_redis.get(track)
            if data is None:
                continue
            try:
                recs = pickle.loads(data)
            except Exception:
                continue
            for r in recs[: self.CANDIDATES_PER_ANCHOR]:
                rid = int(r)
                if rid not in seen_tracks:
                    candidates.add(rid)
            # Bound the size — 10 anchors x 10 = 100, plenty.
        return candidates
