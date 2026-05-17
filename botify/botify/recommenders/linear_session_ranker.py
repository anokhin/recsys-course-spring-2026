import json
import math
import pickle
from collections import Counter, defaultdict

from .recommender import Recommender


class LinearSessionRanker(Recommender):
    """
    Deterministic session-aware linear ranker.

    The model does not deserialize any Python objects produced in a different
    environment. All learned parameters are stored in a plain JSON file.
    """

    def __init__(
        self,
        listen_history_redis,
        sasrec_redis,
        lightfm_redis,
        hstu_redis,
        catalog,
        fallback_recommender,
        weights_path,
    ):
        self.listen_history_redis = listen_history_redis
        self.sasrec_redis = sasrec_redis
        self.lightfm_redis = lightfm_redis
        self.hstu_redis = hstu_redis
        self.catalog = catalog
        self.fallback_recommender = fallback_recommender

        self.track_artist = {int(track.track): track.artist for track in catalog.tracks}
        self.weights = self._load_weights(weights_path)
        self.bias = float(self.weights.pop("bias", 0.0))

        self.max_recent_anchors = int(self.weights.pop("max_recent_anchors", 3))
        max_candidates = self.weights.pop("max_candidates_per_source", {})
        self.max_sasrec = int(max_candidates.get("sasrec", 10))
        self.max_lightfm = int(max_candidates.get("lightfm", 10))
        self.max_hstu = int(max_candidates.get("hstu", 50))

        # Make sure every feature has an explicit weight.
        self.feature_names = [
            "in_sasrec",
            "in_lightfm",
            "in_hstu",
            "sasrec_best_rr",
            "lightfm_best_rr",
            "hstu_best_rr",
            "sasrec_last_rr",
            "lightfm_last_rr",
            "sasrec_second_rr",
            "lightfm_second_rr",
            "source_votes",
            "consensus",
            "same_artist_as_last",
            "artist_seen_count",
            "fresh_artist",
            "anchor_weight_sum",
            "anchor_weight_max",
            "anchor_count",
        ]
        for name in self.feature_names:
            self.weights.setdefault(name, 0.0)

    def _load_weights(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return dict(data)

    def _load_user_history(self, user):
        key = "user:{0}:listens".format(user)
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _load_pickled_list(self, redis_conn, key):
        data = redis_conn.get(key)
        if data is None:
            return []
        values = pickle.loads(data)
        return [int(v) for v in values]

    def _build_recent_anchors(self, history):
        if not history:
            return []

        total_time = defaultdict(float)
        for track, listened_time in history:
            total_time[int(track)] += float(listened_time)

        anchors = []
        used = set()
        unique_idx = 0
        for track, _ in history:
            track = int(track)
            if track in used:
                continue
            used.add(track)
            recency_weight = 1.0 / float(unique_idx + 1)
            strength = math.log1p(total_time[track])
            anchors.append(
                {
                    "track": track,
                    "weight": strength * recency_weight,
                    "is_last": 1 if unique_idx == 0 else 0,
                    "is_second": 1 if unique_idx == 1 else 0,
                }
            )
            unique_idx += 1
            if unique_idx >= self.max_recent_anchors:
                break
        return anchors

    def _accumulate_source_features(self, features, anchor, candidates, source_name):
        for rank, candidate in enumerate(candidates, start=1):
            if candidate not in features:
                continue
            rr = 1.0 / float(rank)

            features[candidate]["in_{0}".format(source_name)] = 1.0
            features[candidate]["{0}_best_rr".format(source_name)] = max(
                features[candidate]["{0}_best_rr".format(source_name)],
                rr,
            )
            if anchor["is_last"]:
                key = "{0}_last_rr".format(source_name)
                features[candidate][key] = max(features[candidate][key], rr)
            if anchor["is_second"]:
                key = "{0}_second_rr".format(source_name)
                features[candidate][key] = max(features[candidate][key], rr)

            features[candidate]["anchor_weight_sum"] += anchor["weight"]
            features[candidate]["anchor_weight_max"] = max(
                features[candidate]["anchor_weight_max"],
                anchor["weight"],
            )
            features[candidate]["anchor_tracks"].add(anchor["track"])

    def _score(self, feature_dict):
        score = self.bias
        for name in self.feature_names:
            score += self.weights[name] * float(feature_dict.get(name, 0.0))
        return score

    def recommend_next(self, user, prev_track, prev_track_time):
        history = self._load_user_history(user)
        if not history:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        seen_tracks = set(int(track) for track, _ in history)
        artist_counts = Counter(
            self.track_artist.get(int(track), "__unknown_artist__")
            for track, _ in history
        )
        anchors = self._build_recent_anchors(history)
        if not anchors:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        last_track = int(history[0][0])
        last_artist = self.track_artist.get(last_track)

        features = defaultdict(
            lambda: {
                "in_sasrec": 0.0,
                "in_lightfm": 0.0,
                "in_hstu": 0.0,
                "sasrec_best_rr": 0.0,
                "lightfm_best_rr": 0.0,
                "hstu_best_rr": 0.0,
                "sasrec_last_rr": 0.0,
                "lightfm_last_rr": 0.0,
                "sasrec_second_rr": 0.0,
                "lightfm_second_rr": 0.0,
                "source_votes": 0.0,
                "consensus": 0.0,
                "same_artist_as_last": 0.0,
                "artist_seen_count": 0.0,
                "fresh_artist": 0.0,
                "anchor_weight_sum": 0.0,
                "anchor_weight_max": 0.0,
                "anchor_count": 0.0,
                "anchor_tracks": set(),
            }
        )

        for anchor in anchors:
            sasrec_candidates = self._load_pickled_list(
                self.sasrec_redis, anchor["track"]
            )[: self.max_sasrec]
            lightfm_candidates = self._load_pickled_list(
                self.lightfm_redis, anchor["track"]
            )[: self.max_lightfm]

            sasrec_candidates = [c for c in sasrec_candidates if c not in seen_tracks]
            lightfm_candidates = [c for c in lightfm_candidates if c not in seen_tracks]

            for candidate in sasrec_candidates:
                _ = features[candidate]
            for candidate in lightfm_candidates:
                _ = features[candidate]

            self._accumulate_source_features(features, anchor, sasrec_candidates, "sasrec")
            self._accumulate_source_features(features, anchor, lightfm_candidates, "lightfm")

        hstu_candidates = self._load_pickled_list(self.hstu_redis, user)[: self.max_hstu]
        hstu_candidates = [c for c in hstu_candidates if c not in seen_tracks]
        for rank, candidate in enumerate(hstu_candidates, start=1):
            _ = features[candidate]
            rr = 1.0 / float(rank)
            features[candidate]["in_hstu"] = 1.0
            features[candidate]["hstu_best_rr"] = max(
                features[candidate]["hstu_best_rr"],
                rr,
            )

        if not features:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        best_candidate = None
        best_score = None

        for candidate, feat in features.items():
            feat["source_votes"] = (
                feat["in_sasrec"] + feat["in_lightfm"] + feat["in_hstu"]
            )
            feat["consensus"] = 1.0 if feat["in_sasrec"] and feat["in_lightfm"] else 0.0
            feat["anchor_count"] = float(len(feat["anchor_tracks"]))

            candidate_artist = self.track_artist.get(candidate)
            if candidate_artist == last_artist:
                feat["same_artist_as_last"] = 1.0

            seen_count = float(artist_counts.get(candidate_artist, 0))
            feat["artist_seen_count"] = seen_count
            feat["fresh_artist"] = 1.0 if seen_count == 0 else 0.0

            score = self._score(feat)
            if best_candidate is None or score > best_score or (
                score == best_score and candidate < best_candidate
            ):
                best_candidate = candidate
                best_score = score

        if best_candidate is None:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)
        return int(best_candidate)
