import json
import math
import pickle
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .recommender import Recommender


class MLHybridRecommender(Recommender):
    """
    Content-based ML recommender with online re-ranking.

    The model is trained only on botify/data/tracks.json:
    each track is represented by a TF-IDF vector over title, genres,
    mood, country, artist metadata and summary. At serving time we build
    a short-term session profile from listened tracks and combine:
      * TF-IDF nearest neighbours of positively consumed tracks;
      * user-level HSTU recommendations, if available;
      * LightFM item-to-item candidates as an ML fallback.

    SasRec-I2I is intentionally not used inside this treatment.
    """

    def __init__(
        self,
        listen_history_redis,
        hstu_recommendations_redis,
        lightfm_recommendations_redis,
        fallback_recommender,
        catalog_path,
        n_neighbors=90,
        hstu_top_k=100,
        lightfm_top_k=10,
    ):
        self.listen_history_redis = listen_history_redis
        self.hstu_recommendations_redis = hstu_recommendations_redis
        self.lightfm_recommendations_redis = lightfm_recommendations_redis
        self.fallback_recommender = fallback_recommender

        self.n_neighbors = n_neighbors
        self.hstu_top_k = hstu_top_k
        self.lightfm_top_k = lightfm_top_k

        self.track_ids = []
        self.id_to_row = {}
        self.artist_by_track = {}
        self.artist_fans_by_track = {}
        self.max_artist_fans = 1.0
        self._neighbor_cache = {}

        self._load_catalog(catalog_path)
        self._fit_content_model()

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)
        artist_counts = Counter(
            self.artist_by_track.get(track)
            for track, _ in history
            if track in self.artist_by_track
        )

        scores = defaultdict(float)

        self._add_hstu_candidates(user, scores, seen_tracks)
        self._add_content_candidates(history, scores, seen_tracks)
        self._add_lightfm_candidates(history, scores, seen_tracks)

        best = self._select_best(user, scores, seen_tracks, artist_counts)
        if best is not None:
            return int(best)

        # Robust fallbacks for cold starts and rare catalog holes.
        best = self._best_content_fallback(user, prev_track, seen_tracks, artist_counts)
        if best is not None:
            return int(best)

        return int(self.fallback_recommender.recommend_next(user, prev_track, prev_track_time))

    def _load_catalog(self, catalog_path):
        documents = []
        with open(catalog_path) as catalog_file:
            for line in catalog_file:
                data = json.loads(line)
                track = int(data["track"])
                self.track_ids.append(track)
                self.id_to_row[track] = len(self.track_ids) - 1
                self.artist_by_track[track] = data.get("artist", "")
                fans = float(data.get("artist_fans") or 0.0)
                self.artist_fans_by_track[track] = fans
                self.max_artist_fans = max(self.max_artist_fans, fans)
                documents.append(self._track_document(data))

        self.documents = documents

    def _track_document(self, data):
        def join(value):
            if isinstance(value, list):
                return " ".join(str(x) for x in value if x is not None)
            return "" if value is None else str(value)

        # Genre/mood fields are repeated deliberately: they are compact and
        # more stable than the long generated summaries.
        parts = [
            join(data.get("title")),
            join(data.get("alternative_title")),
            join(data.get("genres")),
            join(data.get("genres")),
            join(data.get("genres")),
            join(data.get("mood")),
            join(data.get("mood")),
            join(data.get("artist_genre")),
            join(data.get("artist_genre")),
            join(data.get("artist_genres")),
            join(data.get("artist_genres")),
            join(data.get("artist_country")),
            join(data.get("summary")),
            join(data.get("artist")),
        ]
        return " ".join(parts)

    def _fit_content_model(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=30000,
            sublinear_tf=True,
            norm="l2",
        )
        self.track_matrix = self.vectorizer.fit_transform(self.documents)
        self.nn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, len(self.track_ids)),
            metric="cosine",
            algorithm="brute",
        )
        self.nn.fit(self.track_matrix)

    def _load_user_history(self, user):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            track = int(entry["track"])
            track_time = float(entry["time"])
            if track in self.id_to_row:
                history.append((track, track_time))
        return history

    def _loads_pickle(self, raw):
        if raw is None:
            return None
        return pickle.loads(raw)

    def _add_hstu_candidates(self, user, scores, seen_tracks):
        recommendations = self._loads_pickle(self.hstu_recommendations_redis.get(user))
        if not recommendations:
            return

        for rank, track in enumerate(recommendations[: self.hstu_top_k]):
            candidate = int(track)
            if candidate in seen_tracks or candidate not in self.id_to_row:
                continue
            # User-level HSTU is useful, but session context should still dominate.
            scores[candidate] += 0.95 / math.sqrt(rank + 1.0)

    def _add_content_candidates(self, history, scores, seen_tracks):
        # Redis history is newest first. Recent and highly consumed tracks matter more.
        for age, (anchor, listened_time) in enumerate(history[:8]):
            if listened_time <= 0.0:
                continue

            recency = 0.86 ** age
            confidence = 0.20 + min(max(listened_time, 0.0), 1.0)
            anchor_weight = recency * confidence

            for candidate, similarity in self._content_neighbors(anchor):
                if candidate in seen_tracks:
                    continue
                # Similarity is cosine similarity in TF-IDF space.
                scores[candidate] += 2.20 * anchor_weight * max(similarity, 0.0)

    def _add_lightfm_candidates(self, history, scores, seen_tracks):
        for age, (anchor, listened_time) in enumerate(history[:5]):
            if listened_time <= 0.0:
                continue

            raw = self.lightfm_recommendations_redis.get(anchor)
            recommendations = self._loads_pickle(raw)
            if not recommendations:
                continue

            recency = 0.82 ** age
            confidence = 0.10 + min(max(listened_time, 0.0), 1.0)
            for rank, track in enumerate(recommendations[: self.lightfm_top_k]):
                candidate = int(track)
                if candidate in seen_tracks or candidate not in self.id_to_row:
                    continue
                scores[candidate] += 0.75 * recency * confidence / math.sqrt(rank + 1.0)

    def _content_neighbors(self, track):
        if track in self._neighbor_cache:
            return self._neighbor_cache[track]

        row = self.id_to_row.get(track)
        if row is None:
            return []

        distances, indices = self.nn.kneighbors(self.track_matrix[row], return_distance=True)
        result = []
        for distance, idx in zip(distances[0], indices[0]):
            candidate = self.track_ids[int(idx)]
            if candidate == track:
                continue
            similarity = 1.0 - float(distance)
            result.append((candidate, similarity))

        self._neighbor_cache[track] = result
        return result

    def _select_best(self, user, scores, seen_tracks, artist_counts):
        best_track = None
        best_score = None

        for candidate, raw_score in scores.items():
            if candidate in seen_tracks or candidate not in self.id_to_row:
                continue

            artist = self.artist_by_track.get(candidate)
            repeats = artist_counts.get(artist, 0)

            # The simulator discounts repeated artists, so mirror that penalty
            # instead of allowing a single artist to occupy the whole session.
            if repeats >= 4:
                continue
            diversity_multiplier = 0.78 ** repeats

            fans = self.artist_fans_by_track.get(candidate, 0.0)
            popularity_bonus = 0.025 * math.log1p(fans) / math.log1p(self.max_artist_fans)

            final_score = raw_score * diversity_multiplier + popularity_bonus + self._stable_jitter(user, candidate)

            if best_score is None or final_score > best_score:
                best_score = final_score
                best_track = candidate

        return best_track

    def _best_content_fallback(self, user, prev_track, seen_tracks, artist_counts):
        scores = defaultdict(float)
        for candidate, similarity in self._content_neighbors(prev_track):
            if candidate not in seen_tracks:
                scores[candidate] += similarity
        return self._select_best(user, scores, seen_tracks, artist_counts)

    def _stable_jitter(self, user, track):
        # Deterministic tie-breaker: avoids random serving drift between CI runs.
        value = (int(user) * 1000003 + int(track) * 9176 + 17) % 100000
        return value * 1e-10
