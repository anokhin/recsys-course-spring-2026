import json
import pickle
from collections import Counter
from pathlib import Path

from .recommender import Recommender


class SessionBlendRecommender(Recommender):
    def __init__(
        self,
        listen_history_redis,
        sasrec_store,
        lightfm_store,
        hstu_store,
        catalog,
        fallback,
        artifact_path,
    ):
        self.listen_history_redis = listen_history_redis
        self.sasrec_store = sasrec_store
        self.lightfm_store = lightfm_store
        self.hstu_store = hstu_store
        self.catalog = catalog
        self.fallback = fallback
        self.artist_by_track = {track.track: track.artist for track in catalog.tracks}
        payload = pickle.loads(Path(artifact_path).read_bytes())
        self.latents = payload["latent_candidates"]
        self.track_quality = payload["track_quality"]
        self.artist_quality = payload["artist_quality"]
        self.popularity = payload["track_popularity"]
        self.global_candidates = payload["global_candidates"]
        self.global_quality = float(payload["global_quality"])
        self.context_weights = payload["context_weights"]
        self.rank_weights = payload["rank_weights"]
        self.source_bias = payload["source_bias"]
        self.history_limit = int(payload["history_limit"])
        self.source_limits = payload["source_limits"]
        self.min_score = float(payload["min_score"])

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_history(user)
        state_history = (history + [(prev_track, prev_track_time)])[-self.history_limit:]
        seen = {track for track, _ in state_history}
        artists = Counter(self.artist_by_track.get(track) for track, _ in state_history)
        anchor_tracks = self._anchors(prev_track, state_history)
        context_key = self._context_key(prev_track_time, len(state_history))

        candidates = {}
        self._add_candidates(candidates, self.latents.get(user, ()), "latent", "user", seen, self.source_limits["latent"])
        self._add_candidates(candidates, self.hstu_store.get(user, ()), "hstu", "user", seen, self.source_limits["hstu"])
        for anchor_kind, anchor_track in anchor_tracks:
            self._add_candidates(candidates, self.sasrec_store.get(anchor_track, ()), "sasrec", anchor_kind, seen, self.source_limits["sasrec"])
            self._add_candidates(candidates, self.lightfm_store.get(anchor_track, ()), "lightfm", anchor_kind, seen, self.source_limits["lightfm"])
        if len(candidates) < self.source_limits["global"]:
            self._add_candidates(candidates, self.global_candidates, "global", "global", seen, self.source_limits["global"])

        best_track = None
        best_score = None
        for track, meta in candidates.items():
            score = self._score(track, meta, context_key, artists, prev_track)
            if best_score is None or score > best_score:
                best_score = score
                best_track = track

        if best_track is None or (best_score is not None and best_score < self.min_score):
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        return int(best_track)

    def _load_history(self, user: int):
        key = f"user:{user}:listens"
        rows = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for row in rows:
            if isinstance(row, bytes):
                row = row.decode("utf-8")
            data = json.loads(row)
            history.append((int(data["track"]), float(data["time"])))
        history.reverse()
        return history

    def _anchors(self, prev_track, state_history):
        anchors = [("last", prev_track)]
        for track, dwell in sorted(state_history[:-1], key=lambda item: item[1], reverse=True):
            if dwell < 0.65:
                continue
            if all(existing != track for _, existing in anchors):
                anchors.append(("good", track))
            if len(anchors) >= 3:
                break
        return anchors

    def _add_candidates(self, candidates, tracks, source, anchor_kind, seen, limit):
        for rank, track in enumerate(tracks):
            if rank >= limit:
                break
            track = int(track)
            if track in seen:
                continue
            meta = candidates.setdefault(track, {"items": [], "source_count": 0})
            meta["items"].append((source, anchor_kind, rank + 1))
            meta["source_count"] += 1

    def _context_key(self, prev_track_time, history_length):
        if prev_track_time >= 0.9:
            time_bucket = "great"
        elif prev_track_time >= 0.65:
            time_bucket = "good"
        elif prev_track_time >= 0.35:
            time_bucket = "mid"
        else:
            time_bucket = "bad"

        if history_length >= 8:
            depth_bucket = "deep"
        elif history_length >= 4:
            depth_bucket = "mid"
        else:
            depth_bucket = "short"
        return f"{time_bucket}|{depth_bucket}"

    def _score(self, track, meta, context_key, artists, prev_track):
        total = self.global_quality * 0.35
        keys = self.context_weights.get(context_key, {})
        for source, anchor_kind, rank in meta["items"]:
            total += self.source_bias.get(source, 0.0)
            total += keys.get(f"{source}|{anchor_kind}", 0.0)
            total += self.rank_weights.get(source, {}).get(self._rank_bucket(rank), 0.0)
        if meta["source_count"] > 1:
            total += 0.05 * (meta["source_count"] - 1)
        artist = self.artist_by_track.get(track)
        if artist is not None and artists.get(artist, 0) > 0:
            total += 0.06
        if self.artist_by_track.get(prev_track) == artist and artist is not None:
            total += 0.03
        total += 0.18 * self.track_quality.get(track, self.global_quality)
        total += 0.07 * self.artist_quality.get(artist, self.global_quality)
        total += 0.08 * self.popularity.get(track, 0.0)
        return total

    def _rank_bucket(self, rank):
        if rank <= 3:
            return "top3"
        if rank <= 10:
            return "top10"
        if rank <= 30:
            return "top30"
        return "tail"
