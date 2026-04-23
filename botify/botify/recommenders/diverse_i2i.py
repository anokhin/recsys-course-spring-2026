"""Diverse I2I recommender.

Uses an ML-trained item similarity table (`my_i2i.jsonl`, produced by
`script/train_my_i2i.py` via LSI on combined SasRec+LightFM co-occurrences)
and applies per-session artist-aware re-ranking to counter the simulator's
multiplicative artist-repetition penalty.
"""

import json
import pickle
from collections import Counter

from .recommender import Recommender


RECENT_ANCHORS = 5
MIN_ANCHOR_TIME = 0.3
CANDIDATE_POOL = 50
ARTIST_PENALTY = 0.35


class DiverseI2I(Recommender):
    def __init__(self, listen_history_redis, i2i_redis, track_artists, fallback):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.track_artists = track_artists
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_history(user)
        seen = {track for track, _ in history}

        anchors = self._pick_anchors(history, prev_track, prev_track_time)
        artist_counts = self._artist_counts(history)

        for anchor in anchors:
            candidate = self._score_and_pick(anchor, seen, artist_counts)
            if candidate is not None:
                return candidate

        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    def _load_history(self, user: int):
        raw = self.listen_history_redis.lrange(f"user:{user}:listens", 0, -1)
        history = []
        for entry in raw:
            if isinstance(entry, bytes):
                entry = entry.decode("utf-8")
            data = json.loads(entry)
            history.append((int(data["track"]), float(data["time"])))
        return history

    def _pick_anchors(self, history, prev_track, prev_track_time):
        # Anchors are ordered from most recent to oldest; prefer tracks the user
        # actually engaged with (time > threshold) so we stay close to the
        # session's underlying interest rather than its misfires.
        candidates = []
        if prev_track_time >= MIN_ANCHOR_TIME:
            candidates.append(prev_track)
        for track, listened_time in history[:RECENT_ANCHORS]:
            if track in candidates:
                continue
            if listened_time >= MIN_ANCHOR_TIME:
                candidates.append(track)
        if not candidates and history:
            # All listens were weak — fall back to ordering by recency regardless
            for track, _ in history[:RECENT_ANCHORS]:
                if track not in candidates:
                    candidates.append(track)
        return candidates

    def _artist_counts(self, history):
        counts = Counter()
        for track, listened_time in history:
            artist = self.track_artists.get(track)
            if artist is None:
                continue
            # Weight by listened_time so skipped tracks contribute less to the
            # penalty — matches the way the simulator discounts repeats.
            counts[artist] += max(listened_time, 0.1)
        return counts

    def _score_and_pick(self, anchor: int, seen, artist_counts):
        raw = self.i2i_redis.get(anchor)
        if raw is None:
            return None

        candidates = pickle.loads(raw)
        pool = [int(c) for c in candidates[:CANDIDATE_POOL] if int(c) not in seen]
        if not pool:
            return None

        best_track, best_score = None, None
        for rank, track in enumerate(pool):
            base = 1.0 - rank / max(len(pool), 1)
            artist = self.track_artists.get(track)
            penalty = ARTIST_PENALTY * artist_counts.get(artist, 0.0) if artist else 0.0
            score = base - penalty
            if best_score is None or score > best_score:
                best_track, best_score = track, score
        return best_track
