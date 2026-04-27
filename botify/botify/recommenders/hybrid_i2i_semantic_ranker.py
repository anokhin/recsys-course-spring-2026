import json
import math
import pickle
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .recommender import Recommender


class HybridI2ISemanticRanker(Recommender):
    """Context-aware hybrid ranker for HW2.

    The recommender does not use simulator users or simulator embeddings.  It uses
    only Botify-side artifacts available to the service:
      * SasRec-I2I candidates,
      * LightFM-I2I candidates,
      * public track metadata from botify/data/tracks.json.

    Candidate generation is ML-based; online logic is a deterministic lightweight
    ranker over candidates from several models.  The main difference from the
    baseline is that we do not sample one random history anchor.  Instead, we
    score candidates from all recent anchors using the observed listening time
    in the current session and a semantic metadata similarity model.
    """

    def __init__(
        self,
        listen_history_redis,
        sasrec_i2i_redis,
        lightfm_i2i_redis,
        track_redis,
        artists_redis,
        catalog,
        tracks_catalog_path: str,
        fallback_recommender: Recommender,
        history_limit: int = 10,
        max_i2i_per_anchor: int = 10,
        n_components: int = 64,
    ):
        self.listen_history_redis = listen_history_redis
        self.sasrec_i2i_redis = sasrec_i2i_redis
        self.lightfm_i2i_redis = lightfm_i2i_redis
        self.track_redis = track_redis
        self.artists_redis = artists_redis
        self.catalog = catalog
        self.tracks_catalog_path = tracks_catalog_path
        self.fallback_recommender = fallback_recommender
        self.history_limit = history_limit
        self.max_i2i_per_anchor = max_i2i_per_anchor
        self.n_components = n_components

        self._artist_cache: Dict[int, Optional[str]] = {}
        self._track_to_idx: Dict[int, int] = {}
        self._idx_to_track: List[int] = []
        self._artist_by_idx: List[str] = []
        self._popularity_prior: np.ndarray = np.zeros(1, dtype=np.float32)
        self._embeddings: np.ndarray = np.zeros((1, 1), dtype=np.float32)

        self._fit_semantic_model()

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        if not history:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        seen_tracks = {track for track, _ in history}
        artist_counts = Counter(
            artist for artist in (self._artist(track) for track, _ in history)
            if artist is not None
        )

        candidate_scores = defaultdict(float)

        # Newest item is first in Redis.  Use all recent anchors, but make the
        # impact of an anchor depend on both recency and observed listening time.
        for pos, (anchor, listened_time) in enumerate(history[: self.history_limit]):
            anchor_weight = self._anchor_weight(pos, listened_time)

            # SasRec stays an important candidate generator, but it is no longer
            # used as a stochastic one-anchor policy.
            self._add_i2i_candidates(
                candidate_scores,
                self.sasrec_i2i_redis,
                anchor,
                seen_tracks,
                source_weight=2.25 * anchor_weight,
                rank_decay=0.18,
            )

            # LightFM brings a second collaborative signal, useful when SasRec's
            # first candidates are already seen or too narrow for the session.
            self._add_i2i_candidates(
                candidate_scores,
                self.lightfm_i2i_redis,
                anchor,
                seen_tracks,
                source_weight=1.15 * anchor_weight,
                rank_decay=0.22,
            )

        if not candidate_scores:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        profile = self._session_profile(history)
        best_track = None
        best_score = float("-inf")

        for track, base_score in candidate_scores.items():
            if track in seen_tracks:
                continue

            idx = self._track_to_idx.get(track)
            if idx is None:
                semantic_score = 0.0
                popularity = 0.0
            else:
                semantic_score = float(np.dot(self._embeddings[idx], profile))
                popularity = float(self._popularity_prior[idx])

            artist = self._artist(track)
            repeat_penalty = 0.0
            if artist is not None:
                # The simulator discounts repeated artists, so repeated artists
                # are allowed only when the candidate score is clearly strong.
                repeat_penalty = 0.42 * artist_counts.get(artist, 0)

            # Source score gives stability; semantic score chooses the candidate
            # that best matches the current session intent.
            score = (
                base_score
                + 1.35 * semantic_score
                + 0.08 * popularity
                - repeat_penalty
            )

            if score > best_score or (score == best_score and (best_track is None or track < best_track)):
                best_score = score
                best_track = track

        if best_track is not None:
            return int(best_track)

        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

    @staticmethod
    def _anchor_weight(position: int, listened_time: float) -> float:
        listened_time = max(0.0, min(float(listened_time), 1.0))
        # Low listening time is a negative signal; high listening time means
        # that this anchor probably matches the current session interest.
        quality = 0.05 + listened_time ** 1.7
        recency = 0.88 ** position
        return quality * recency

    def _add_i2i_candidates(
        self,
        scores,
        redis_conn,
        anchor: int,
        seen_tracks,
        source_weight: float,
        rank_decay: float,
    ) -> None:
        raw = redis_conn.get(anchor)
        if raw is None:
            return

        try:
            recommendations = list(self.catalog.from_bytes(raw))
        except Exception:
            try:
                recommendations = list(pickle.loads(raw))
            except Exception:
                return

        for rank, candidate in enumerate(recommendations[: self.max_i2i_per_anchor]):
            candidate = int(candidate)
            if candidate in seen_tracks:
                continue
            scores[candidate] += source_weight / (1.0 + rank_decay * rank)

    def _load_user_history(self, user: int) -> List[Tuple[int, float]]:
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, self.history_limit - 1)

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

    def _session_profile(self, history: List[Tuple[int, float]]) -> np.ndarray:
        vectors = []
        weights = []
        for pos, (track, listened_time) in enumerate(history[: self.history_limit]):
            idx = self._track_to_idx.get(track)
            if idx is None:
                continue
            vectors.append(self._embeddings[idx])
            weights.append(self._anchor_weight(pos, listened_time))

        if not vectors:
            return self._embeddings[0]

        weights = np.asarray(weights, dtype=np.float32)
        matrix = np.vstack(vectors)
        profile = np.average(matrix, axis=0, weights=weights)
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
        return profile.astype(np.float32)

    def _fit_semantic_model(self) -> None:
        records = []
        with open(self.tracks_catalog_path, encoding="utf-8") as tracks_file:
            for line in tracks_file:
                if line.strip():
                    records.append(json.loads(line))

        records.sort(key=lambda r: int(r["track"]))

        self._idx_to_track = [int(record["track"]) for record in records]
        self._track_to_idx = {track: idx for idx, track in enumerate(self._idx_to_track)}
        self._artist_by_idx = [str(record.get("artist", "")) for record in records]

        docs = [self._record_to_text(record) for record in records]
        vectorizer = TfidfVectorizer(
            max_features=60000,
            min_df=1,
            ngram_range=(1, 2),
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
        )
        x = vectorizer.fit_transform(docs)

        n_components = min(self.n_components, max(2, min(x.shape) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        emb = svd.fit_transform(x)
        self._embeddings = normalize(emb).astype(np.float32)

        fans = np.array([self._safe_float(r.get("artist_fans", 1.0)) for r in records], dtype=np.float32)
        fans = np.log1p(np.maximum(fans, 0.0))
        fans = (fans - fans.min()) / (fans.max() - fans.min() + 1e-9)
        self._popularity_prior = fans.astype(np.float32)

    @staticmethod
    def _record_to_text(record: dict) -> str:
        def join(value):
            if isinstance(value, list):
                return " ".join(str(x) for x in value)
            return "" if value is None else str(value)

        parts = [
            record.get("title", ""),
            record.get("artist", ""),
            join(record.get("genres", [])),
            join(record.get("artist_genres", [])),
            record.get("artist_genre", ""),
            record.get("artist_country", ""),
            record.get("mood", ""),
            str(record.get("year", "")),
            record.get("summary", ""),
        ]
        return " ".join(str(p) for p in parts if p is not None)

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            if isinstance(value, str):
                value = value.split("-")[0]
            return float(value)
        except Exception:
            return default

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
