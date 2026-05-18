import json
import math
import pickle
from collections import Counter, defaultdict

from .recommender import Recommender


class MLHybridRecommender(Recommender):
    """
    Fast online ML-style ranker for HW2.

    The treatment is not a direct SasRec-I2I policy: it builds a candidate pool
    from several trained recommenders (SasRec-I2I, LightFM-I2I, HSTU user lists)
    and re-ranks candidates with implicit feedback from the current user's
    listening history.  The scorer mirrors the target metric: it rewards
    candidates supported by high-time anchors and discounts tracks by artists
    already repeated in the current session/history.

    All state used at serving time comes from botify/data and Redis listen logs;
    sim/data is never read.
    """

    def __init__(
        self,
        listen_history_redis,
        sasrec_recommendations_redis,
        lightfm_recommendations_redis,
        hstu_recommendations_redis,
        fallback_recommender,
        catalog_path,
        history_limit=10,
    ):
        self.listen_history_redis = listen_history_redis
        self.sasrec_recommendations_redis = sasrec_recommendations_redis
        self.lightfm_recommendations_redis = lightfm_recommendations_redis
        self.hstu_recommendations_redis = hstu_recommendations_redis
        self.fallback_recommender = fallback_recommender
        self.history_limit = history_limit

        self.artist_by_track = {}
        self.genre_by_track = {}
        self.mood_by_track = {}
        self.country_by_track = {}
        self.fans_by_track = {}
        self.max_fans = 1.0
        self.track_ids = []

        self._load_catalog(catalog_path)

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        if not history:
            return int(self.fallback_recommender.recommend_next(user, prev_track, prev_track_time))

        seen_tracks = {track for track, _ in history}
        artist_counts = Counter(self.artist_by_track.get(track) for track, _ in history)
        recent_good_artists = self._recent_good_artists(history)
        recent_bad_artists = self._recent_bad_artists(history)

        scores = defaultdict(float)
        reasons = defaultdict(set)

        self._score_i2i_source(
            scores,
            reasons,
            history,
            seen_tracks,
            self.sasrec_recommendations_redis,
            source_weight=1.00,
            source_name="sasrec",
            max_rank=10,
        )
        self._score_i2i_source(
            scores,
            reasons,
            history,
            seen_tracks,
            self.lightfm_recommendations_redis,
            source_weight=0.22,
            source_name="lightfm",
            max_rank=6,
        )
        self._score_hstu_source(
            scores,
            reasons,
            user,
            history,
            seen_tracks,
            source_weight=0.18,
            max_rank=80,
        )

        best = self._select_best(
            user=user,
            candidates=scores,
            reasons=reasons,
            seen_tracks=seen_tracks,
            artist_counts=artist_counts,
            recent_good_artists=recent_good_artists,
            recent_bad_artists=recent_bad_artists,
        )
        if best is not None:
            return int(best)

        # Last robust fallback: direct trained I2I candidate from the latest item,
        # filtered by seen track only.
        best = self._first_unseen_from_redis(self.sasrec_recommendations_redis, prev_track, seen_tracks)
        if best is not None:
            return int(best)

        return int(self.fallback_recommender.recommend_next(user, prev_track, prev_track_time))

    def _load_catalog(self, catalog_path):
        with open(catalog_path) as catalog_file:
            for line in catalog_file:
                data = json.loads(line)
                track = int(data["track"])
                self.track_ids.append(track)
                self.artist_by_track[track] = data.get("artist") or ""

                genres = data.get("genres") or []
                artist_genres = data.get("artist_genres") or []
                all_genres = list(genres) + list(artist_genres)
                self.genre_by_track[track] = tuple(str(g).lower() for g in all_genres)

                self.mood_by_track[track] = str(data.get("mood") or "").lower()
                self.country_by_track[track] = self._norm_country(data.get("artist_country") or "")
                fans = float(data.get("artist_fans") or 0.0)
                self.fans_by_track[track] = fans
                self.max_fans = max(self.max_fans, fans)

    def _load_user_history(self, user):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, self.history_limit - 1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                entry = json.loads(raw)
                track = int(entry["track"])
                listened_time = float(entry["time"])
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if track in self.artist_by_track:
                history.append((track, listened_time))
        return history

    def _score_i2i_source(
        self,
        scores,
        reasons,
        history,
        seen_tracks,
        recommendations_redis,
        source_weight,
        source_name,
        max_rank,
    ):
        # Redis history is newest first.  We aggregate over anchors instead of
        # sampling one anchor randomly, so the treatment has lower serving noise
        # than the baseline I2I policy.
        for age, (anchor, listened_time) in enumerate(history[: self.history_limit]):
            if listened_time <= 0.0:
                continue

            recommendations = self._loads_pickle(recommendations_redis.get(anchor))
            if not recommendations:
                continue

            # High-time anchors should dominate.  The first track of the session
            # arrives with time=1.0, which is a strong signal of the hidden
            # session interest in the simulator.
            confidence = 0.15 + min(max(listened_time, 0.0), 1.0)
            recency = 0.88 ** age
            anchor_weight = source_weight * confidence * recency

            for rank, track in enumerate(recommendations[:max_rank]):
                candidate = int(track)
                if candidate in seen_tracks or candidate not in self.artist_by_track:
                    continue
                scores[candidate] += anchor_weight / math.log2(rank + 2.0)
                reasons[candidate].add(source_name)

    def _score_hstu_source(self, scores, reasons, user, history, seen_tracks, source_weight, max_rank):
        recommendations = self._loads_pickle(self.hstu_recommendations_redis.get(user))
        if not recommendations:
            return

        liked_profile = self._liked_profile(history)
        for rank, track in enumerate(recommendations[:max_rank]):
            candidate = int(track)
            if candidate in seen_tracks or candidate not in self.artist_by_track:
                continue

            profile_match = self._profile_match(candidate, liked_profile)
            # HSTU is user-level rather than session-level, so use it mainly as
            # a tie-breaker / extra support unless it matches recent liked items.
            scores[candidate] += source_weight * (0.35 + profile_match) / math.sqrt(rank + 1.0)
            reasons[candidate].add("hstu")

    def _select_best(
        self,
        user,
        candidates,
        reasons,
        seen_tracks,
        artist_counts,
        recent_good_artists,
        recent_bad_artists,
    ):
        best_track = None
        best_score = None

        for candidate, raw_score in candidates.items():
            if candidate in seen_tracks or candidate not in self.artist_by_track:
                continue

            artist = self.artist_by_track.get(candidate)
            repeats = artist_counts.get(artist, 0)

            # The simulator applies artist_discount_gamma ** repeats.  Penalize
            # repeated artists a little stronger than 0.8 to avoid chains of the
            # same artist, a common I2I failure mode for music.
            artist_multiplier = 0.66 ** repeats
            if repeats >= 3:
                artist_multiplier *= 0.25

            # Give a mild bonus when a non-repeated artist has just worked well.
            # This handles artists appearing under alternative names / covers.
            if artist in recent_good_artists and repeats == 0:
                artist_multiplier *= 1.07

            if artist in recent_bad_artists:
                artist_multiplier *= 0.72

            # If several trained models agree, the candidate is more robust.
            agreement_bonus = 1.0 + 0.08 * max(0, len(reasons[candidate]) - 1)

            # Stable, tiny popularity prior: helps cold/rank ties but cannot
            # dominate model evidence.
            fans = self.fans_by_track.get(candidate, 0.0)
            popularity_bonus = 0.012 * math.log1p(fans) / math.log1p(self.max_fans)

            score = raw_score * artist_multiplier * agreement_bonus + popularity_bonus
            score += self._stable_jitter(user, candidate)

            if best_score is None or score > best_score:
                best_score = score
                best_track = candidate

        return best_track

    def _liked_profile(self, history):
        genres = Counter()
        moods = Counter()
        countries = Counter()
        artists = Counter()

        for age, (track, listened_time) in enumerate(history[: self.history_limit]):
            if listened_time <= 0.25:
                continue
            weight = (0.9 ** age) * min(max(listened_time, 0.0), 1.0)
            for genre in self.genre_by_track.get(track, ()):
                genres[genre] += weight
            moods[self.mood_by_track.get(track, "")] += weight
            countries[self.country_by_track.get(track, "")] += weight
            artists[self.artist_by_track.get(track, "")] += weight

        return {
            "genres": genres,
            "moods": moods,
            "countries": countries,
            "artists": artists,
        }

    def _profile_match(self, candidate, profile):
        score = 0.0

        for genre in self.genre_by_track.get(candidate, ()):
            score += 0.16 * min(profile["genres"].get(genre, 0.0), 2.0)

        score += 0.10 * min(profile["moods"].get(self.mood_by_track.get(candidate, ""), 0.0), 2.0)
        score += 0.08 * min(profile["countries"].get(self.country_by_track.get(candidate, ""), 0.0), 2.0)

        # Same artist is useful as a relevance hint, but repeated-artist penalty
        # is applied later, so keep this bonus bounded.
        score += 0.08 * min(profile["artists"].get(self.artist_by_track.get(candidate, ""), 0.0), 1.0)
        return min(score, 1.0)

    def _recent_good_artists(self, history):
        return {
            self.artist_by_track.get(track)
            for track, listened_time in history[:4]
            if listened_time >= 0.75
        }

    def _recent_bad_artists(self, history):
        return {
            self.artist_by_track.get(track)
            for track, listened_time in history[:4]
            if listened_time <= 0.20
        }

    def _first_unseen_from_redis(self, recommendations_redis, anchor, seen_tracks):
        recommendations = self._loads_pickle(recommendations_redis.get(anchor))
        if not recommendations:
            return None
        for track in recommendations:
            candidate = int(track)
            if candidate not in seen_tracks:
                return candidate
        return None

    def _loads_pickle(self, raw):
        if raw is None:
            return None
        try:
            return pickle.loads(raw)
        except Exception:
            return None

    def _norm_country(self, country):
        country = str(country).strip().lower()
        if country in {"usa", "united states", "us", "u.s.", "u.s.a."}:
            return "usa"
        if country in {"uk", "united kingdom", "england"}:
            return "uk"
        return country

    def _stable_jitter(self, user, track):
        # Deterministic tie-breaker: prevents serving drift without using random.
        value = (int(user) * 1000003 + int(track) * 9176 + 17) % 100000
        return value * 1e-10
