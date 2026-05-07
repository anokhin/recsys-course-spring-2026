import random
from collections import defaultdict, Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class MLRankerRecommender:
    def __init__(self, top_k=20, candidate_pool_size=100, random_state=42):
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )

        self.track_popularity = Counter()
        self.track_listen_sum = defaultdict(float)
        self.track_listen_cnt = defaultdict(int)
        self.user_history = defaultdict(list)
        self.user_track_cnt = defaultdict(Counter)
        self.covisitation = defaultdict(Counter)

        self.popular_tracks = []
        self.is_fitted = False

    def fit(self, logs):
        sessions = defaultdict(list)

        for row in logs:
            user_id = row["user_id"]
            track_id = row["track_id"]
            session_id = row["session_id"]
            listen_time = float(row.get("listen_time", 0.0))

            self.track_popularity[track_id] += 1
            self.track_listen_sum[track_id] += listen_time
            self.track_listen_cnt[track_id] += 1

            self.user_history[user_id].append(track_id)
            self.user_track_cnt[user_id][track_id] += 1
            sessions[session_id].append(track_id)

        self.popular_tracks = [
            track for track, _ in self.track_popularity.most_common(500)
        ]

        for _, tracks in sessions.items():
            unique_tracks = list(dict.fromkeys(tracks))
            for i, t1 in enumerate(unique_tracks):
                for t2 in unique_tracks[max(0, i - 10): i + 11]:
                    if t1 != t2:
                        self.covisitation[t1][t2] += 1

        X, y = self._build_training_data(sessions)

        if len(X) > 0 and len(set(y)) > 1:
            self.model.fit(X, y)
            self.is_fitted = True

        return self

    def recommend(self, user_id, n_items=None):
        if n_items is None:
            n_items = self.top_k

        candidates = self._generate_candidates(user_id)

        if not candidates:
            return self.popular_tracks[:n_items]

        features = np.array([
            self._make_features(user_id, track_id)
            for track_id in candidates
        ])

        if self.is_fitted:
            scores = self.model.predict_proba(features)[:, 1]
        else:
            scores = np.array([
                self.track_popularity.get(track_id, 0)
                for track_id in candidates
            ])

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        result = []
        seen = set(self.user_history.get(user_id, [])[-20:])

        for track_id, _ in ranked:
            if track_id not in result and track_id not in seen:
                result.append(track_id)
            if len(result) >= n_items:
                break

        if len(result) < n_items:
            for track_id in self.popular_tracks:
                if track_id not in result and track_id not in seen:
                    result.append(track_id)
                if len(result) >= n_items:
                    break

        return result

    def _generate_candidates(self, user_id):
        candidates = set()
        history = self.user_history.get(user_id, [])

        for track_id in history[-10:]:
            for candidate, _ in self.covisitation[track_id].most_common(30):
                candidates.add(candidate)

        for track_id, _ in self.user_track_cnt[user_id].most_common(20):
            candidates.add(track_id)

        for track_id in self.popular_tracks[:100]:
            candidates.add(track_id)

        candidates = list(candidates)

        if len(candidates) > self.candidate_pool_size:
            random.Random(self.random_state).shuffle(candidates)
            candidates = candidates[:self.candidate_pool_size]

        return candidates

    def _make_features(self, user_id, track_id):
        popularity = self.track_popularity.get(track_id, 0)
        listen_cnt = self.track_listen_cnt.get(track_id, 0)

        mean_listen_time = (
            self.track_listen_sum[track_id] / listen_cnt
            if listen_cnt > 0 else 0.0
        )

        user_track_count = self.user_track_cnt[user_id].get(track_id, 0)

        history = self.user_history.get(user_id, [])
        recent_history = history[-10:]

        covisit_score = 0.0
        for recent_track in recent_history:
            covisit_score += self.covisitation[recent_track].get(track_id, 0)

        already_seen = 1.0 if track_id in history else 0.0

        return [
            np.log1p(popularity),
            mean_listen_time,
            user_track_count,
            np.log1p(covisit_score),
            already_seen,
            len(history),
        ]

    def _build_training_data(self, sessions):
        X = []
        y = []

        for _, tracks in sessions.items():
            if len(tracks) < 2:
                continue

            for i in range(len(tracks) - 1):
                positive_track = tracks[i + 1]

                fake_user = "__training_user__"
                self.user_history[fake_user] = tracks[: i + 1]

                X.append(
                    self._make_features(fake_user, positive_track)
                )
                y.append(1)

                negatives = self._sample_negative_tracks(
                    exclude=set(tracks),
                    n=2
                )

                for negative_track in negatives:
                    X.append(
                        self._make_features(fake_user, negative_track)
                    )
                    y.append(0)

        return np.array(X), np.array(y)

    def _sample_negative_tracks(self, exclude, n=2):
        negatives = []

        for track_id in self.popular_tracks:
            if track_id not in exclude:
                negatives.append(track_id)
            if len(negatives) >= n:
                break

        return negatives
