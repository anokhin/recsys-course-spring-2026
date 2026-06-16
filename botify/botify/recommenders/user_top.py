import json
import pickle
from collections import Counter

from .recommender import Recommender


class UserTop(Recommender):
    def __init__(self, listen_history_redis, recommendations_redis, sasrec_redis, lightfm_redis, catalog, fallback, model_path):
        self.listen_history_redis = listen_history_redis
        self.recommendations_redis = recommendations_redis
        self.sasrec_redis = sasrec_redis
        self.lightfm_redis = lightfm_redis
        self.catalog = catalog
        self.fallback = fallback
        self.artist_cache = {track.track: track.artist for track in catalog.tracks}
        with open(model_path, encoding="utf-8") as f:
            self.model = json.load(f)

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_history(user)
        seen = {track for track, _ in history}
        artists = Counter()
        for track, _ in history:
            artist = self._artist(track)
            if artist is not None:
                artists[artist] += 1

        candidates = self._session_candidates(user, history, prev_track)
        best_track = None
        best_score = -1.0

        for track, info in candidates.items():
            if track in seen:
                continue
            score = self._score(track, info, prev_track_time, artists)
            if score > best_score:
                best_score = score
                best_track = track

        if best_track is not None and best_score >= self.model.get("min_score", 0.0):
            return best_track

        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    def _load_history(self, user: int):
        key = f"user:{user}:listens"
        rows = self.listen_history_redis.lrange(key, 0, -1)
        result = []
        for row in rows:
            if isinstance(row, bytes):
                row = row.decode("utf-8")
            data = json.loads(row)
            result.append((int(data["track"]), float(data["time"])))
        return result

    def _session_candidates(self, user, history, prev_track):
        anchors = [prev_track]
        for track, score in sorted(history, key=lambda x: x[1], reverse=True):
            if score >= 0.75 and track not in anchors:
                anchors.append(track)
            if len(anchors) >= 4:
                break

        result = {}
        for anchor_pos, anchor in enumerate(anchors):
            for name, redis in (("sasrec", self.sasrec_redis), ("lightfm", self.lightfm_redis)):
                raw = redis.get(anchor)
                if raw is None:
                    continue
                for rank, track in enumerate(pickle.loads(raw)[:30]):
                    track = int(track)
                    info = result.setdefault(track, {"sources": set(), "best_rank": 1000, "anchor_pos": 1000, "user_rank": 1000})
                    info["sources"].add(name)
                    info["best_rank"] = min(info["best_rank"], rank)
                    info["anchor_pos"] = min(info["anchor_pos"], anchor_pos)

        raw_recs = self.recommendations_redis.get(user)
        if raw_recs is not None:
            for rank, track in enumerate(pickle.loads(raw_recs)[:50]):
                track = int(track)
                info = result.setdefault(track, {"sources": set(), "best_rank": 1000, "anchor_pos": 1000, "user_rank": 1000})
                info["sources"].add("user")
                info["user_rank"] = min(info["user_rank"], rank)

        return result

    def _score(self, track, info, prev_time, artists):
        artist = self._artist(track)
        source_key = "+".join(sorted(info["sources"]))
        keys = [
            "prev_time=" + self._time_bucket(prev_time),
            "source=" + source_key,
            "rank=" + self._rank_bucket(info["best_rank"]),
            "anchor=" + self._anchor_bucket(info["anchor_pos"]),
            "user_rank=" + self._rank_bucket(info["user_rank"]),
            "artist_seen=" + self._artist_bucket(artists.get(artist, 0)),
            "track_prior=" + self._prior_bucket(track),
            "artist_prior=" + self._artist_prior_bucket(artist),
        ]

        total = self.model["global_mean"]
        weight = 1.0
        values = self.model["values"]
        for key in keys:
            data = values.get(key)
            if data is None:
                continue
            w = min(float(data["count"]) / 200.0, 1.5)
            total += float(data["mean"]) * w
            weight += w
        return total / weight

    def _time_bucket(self, value):
        if value < 0.35:
            return "bad"
        if value < 0.65:
            return "mid"
        if value < 0.9:
            return "good"
        return "great"

    def _rank_bucket(self, rank):
        if rank <= 2:
            return "top3"
        if rank <= 9:
            return "top10"
        if rank <= 29:
            return "top30"
        return "none"

    def _anchor_bucket(self, pos):
        if pos == 0:
            return "last"
        if pos <= 3:
            return "session"
        return "none"

    def _artist_bucket(self, count):
        if count == 0:
            return "new"
        if count == 1:
            return "once"
        return "repeat"

    def _prior_bucket(self, track):
        mean = self.model.get("track_mean", {}).get(str(track), self.model["global_mean"])
        return self._mean_bucket(mean)

    def _artist_prior_bucket(self, artist):
        mean = self.model.get("artist_mean", {}).get(artist, self.model["global_mean"])
        return self._mean_bucket(mean)

    def _mean_bucket(self, mean):
        if mean < 0.35:
            return "low"
        if mean < 0.55:
            return "mid"
        if mean < 0.75:
            return "high"
        return "very_high"

    def _artist(self, track):
        return self.artist_cache.get(track)
