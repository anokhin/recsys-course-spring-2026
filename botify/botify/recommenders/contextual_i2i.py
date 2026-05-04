import json
import random
from collections import Counter, defaultdict

from .recommender import Recommender


class ContextualI2IRecommender(Recommender):
    def __init__(
        self,
        listen_history_redis,
        combined_i2i_redis,
        sasrec_i2i_redis,
        tracks_redis,
        artists_redis,
        catalog,
        hstu_redis,
        fallback_recommender,
    ):
        self.listen_history_redis = listen_history_redis
        self.combined_i2i_redis = combined_i2i_redis
        self.sasrec_i2i_redis = sasrec_i2i_redis
        self.tracks_redis = tracks_redis
        self.artists_redis = artists_redis
        self.catalog = catalog
        self.hstu_redis = hstu_redis
        self.fallback = fallback_recommender

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(t for t, _ in history)
        seen_tracks.add(prev_track)

        artist_counts = self._count_artists(history, prev_track)

        candidate = self._recommend_from_context(
            prev_track, prev_track_time, seen_tracks, artist_counts
        )
        if candidate is not None:
            return candidate

        candidate = self._recommend_from_history(history, seen_tracks, artist_counts)
        if candidate is not None:
            return candidate

        candidate = self._recommend_from_hstu(user, seen_tracks, artist_counts)
        if candidate is not None:
            return candidate

        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    def _count_artists(self, history, prev_track):
        counts = Counter()
        for track_id, _ in history:
            artist = self._get_artist(track_id)
            if artist:
                counts[artist] += 1
        artist = self._get_artist(prev_track)
        if artist:
            counts[artist] += 1
        return counts

    def _get_artist(self, track_id):
        data = self.tracks_redis.get(track_id)
        if data is None:
            return None
        track = self.catalog.from_bytes(data)
        return track.artist if hasattr(track, "artist") else None

    def _recommend_from_context(self, anchor, prev_time, seen_tracks, artist_counts):
        if prev_time < 0.4:
            return None

        recs = self._get_i2i_candidates(anchor, self.combined_i2i_redis)
        if not recs:
            recs = self._get_i2i_candidates(anchor, self.sasrec_i2i_redis)
        if not recs:
            return None

        return self._pick_best(recs, seen_tracks, artist_counts)

    def _recommend_from_history(self, history, seen_tracks, artist_counts):
        if not history:
            return None

        track_time = defaultdict(float)
        for track, listened_time in history:
            track_time[track] += listened_time

        anchors = list(track_time.keys())
        weights = [track_time[t] for t in anchors]

        tried = set()
        while anchors and len(tried) < len(anchors):
            anchor = random.choices(anchors, weights=weights, k=1)[0]
            if anchor in tried:
                tried.add(anchor)
                anchors_subset = [a for a in anchors if a not in tried]
                if not anchors_subset:
                    break
                continue
            tried.add(anchor)

            recs = self._get_i2i_candidates(anchor, self.combined_i2i_redis)
            if not recs:
                recs = self._get_i2i_candidates(anchor, self.sasrec_i2i_redis)
            if not recs:
                continue

            candidate = self._pick_best(recs, seen_tracks, artist_counts)
            if candidate is not None:
                return candidate

        return None

    def _recommend_from_hstu(self, user, seen_tracks, artist_counts):
        data = self.hstu_redis.get(user)
        if data is None:
            return None

        recs = list(self.catalog.from_bytes(data))
        random.shuffle(recs)
        return self._pick_best(recs, seen_tracks, artist_counts)

    def _pick_best(self, candidates, seen_tracks, artist_counts, max_artist_count=2):
        for candidate in candidates:
            cid = int(candidate)
            if cid in seen_tracks:
                continue
            artist = self._get_artist(cid)
            if artist and artist_counts.get(artist, 0) >= max_artist_count:
                continue
            return cid
        return None

    def _get_i2i_candidates(self, anchor, i2i_redis):
        import pickle
        data = i2i_redis.get(anchor)
        if data is None:
            return None
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
