import json
import pickle
import random
from collections import Counter, defaultdict

from .recommender import Recommender


class DiverseI2IRecommender(Recommender):
    """I2I с учётом истории и штрафом за повтор артиста в сессии.

    Отличие от базового I2IRecommender:
    - якоря сэмплируются с весом time из listen history;
    - кандидаты пропускаются, если артист уже часто встречался в сессии;
    - порядок кандидатов в I2I-списке сохраняется (без shuffle).
    """

    def __init__(
        self,
        listen_history_redis,
        i2i_redis,
        tracks_redis,
        catalog,
        fallback_recommender,
        max_artist_repeats: int = 1,
    ):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.tracks_redis = tracks_redis
        self.catalog = catalog
        self.fallback_recommender = fallback_recommender
        self.max_artist_repeats = max_artist_repeats

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = {track for track, _ in history}
        seen_tracks.add(prev_track)

        artist_counter = Counter()
        for track, _ in history:
            artist = self._artist_of(track)
            if artist is not None:
                artist_counter[artist] += 1
        prev_artist = self._artist_of(prev_track)
        if prev_artist is not None:
            artist_counter[prev_artist] += 1

        if history:
            track_time = defaultdict(float)
            for track, listened_time in history:
                track_time[track] += listened_time

            anchors = list(track_time.keys())
            weights = [track_time[track] for track in anchors]

            while anchors:
                anchor = random.choices(anchors, weights=weights, k=1)[0]
                candidate = self._recommend_from_anchor(
                    anchor, seen_tracks, artist_counter
                )
                if candidate is not None:
                    return candidate

                idx = anchors.index(anchor)
                anchors.pop(idx)
                weights.pop(idx)

        # Холодный старт / пустая история — пробуем якорь = prev_track
        candidate = self._recommend_from_anchor(prev_track, seen_tracks, artist_counter)
        if candidate is not None:
            return candidate

        return self.fallback_recommender.recommend_next(
            user, prev_track, prev_track_time
        )

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

    def _artist_of(self, track_id: int):
        data = self.tracks_redis.get(track_id)
        if data is None:
            return None
        return self.catalog.from_bytes(data).artist

    def _recommend_from_anchor(self, anchor: int, seen_tracks, artist_counter):
        data = self.i2i_redis.get(anchor)
        if data is None:
            return None

        recommendations = pickle.loads(data)
        # 1) сначала кандидаты с новым артистом
        # 2) потом любые непрослушанные
        deferred = []
        for track in recommendations:
            candidate = int(track)
            if candidate in seen_tracks:
                continue
            artist = self._artist_of(candidate)
            if artist is not None and artist_counter[artist] >= self.max_artist_repeats:
                deferred.append(candidate)
                continue
            return candidate

        for candidate in deferred:
            if candidate not in seen_tracks:
                return candidate
        return None
