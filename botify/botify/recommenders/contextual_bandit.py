import json
import numpy as np
from collections import defaultdict

from .recommender import Recommender


class ContextualBanditRecommender(Recommender):
    """
    LinUCB recommender with contextual features from tracks metadata.
    
    Learns online from user feedback (listen time as reward).
    Balances exploration (trying new tracks) vs exploitation (playing known good tracks).
    """
    
    def __init__(self, listen_history_redis, track_redis, catalog, fallback_recommender, alpha=1.0):
        self.listen_history_redis = listen_history_redis
        self.track_redis = track_redis
        self.catalog = catalog
        self.fallback = fallback_recommender
        self.alpha = alpha
        self._load_pretrained()
        
        # Cache for track features
        self._track_features_cache = {}
        
        # LinUCB parameters per user
        self.A = {}  # user_id -> matrix (d x d)
        self.b = {}  # user_id -> vector (d,)
        
        # Pre-define genre list for feature extraction
        self.all_genres = [
            "Pop", "Rock", "Disco", "Soft Rock", "Funk", "Soul", 
            "Blues", "Europop", "Country", "Jazz", "R&B", "Hip-Hop",
            "Electronic", "Folk", "Metal", "Punk", "Reggae", "Classical"
        ]
        self.all_moods = ["Energetic", "Romantic", "Nostalgic", "Sad", "Melancholic"]
        
        # Feature dimension
        self.d = len(self.all_genres) + 1 + len(self.all_moods) + 1
    
    def _get_track_info(self, track_id):
        """Get track metadata from Redis."""
        track_bytes = self.track_redis.get(track_id)
        if track_bytes is None:
            return None
        return self.catalog.from_bytes(track_bytes)
    
    def _extract_features(self, track) -> np.ndarray:
        """Convert track metadata to feature vector."""
        if track is None:
            return np.zeros(self.d)
        
        features = []
        
        # Genre features (multi-hot: track genres + artist genre)
        track_genres = set(getattr(track, 'genres', []))
        artist_genre = getattr(track, 'artist_genre', '')
        for genre in self.all_genres:
            features.append(1.0 if genre in track_genres or genre == artist_genre else 0.0)
        
        # Year normalized
        year = getattr(track, 'year', 1970)
        features.append(year / 2020.0)
        
        # Mood (one-hot, if available)
        track_mood = getattr(track, 'mood', '')
        for mood in self.all_moods:
            features.append(1.0 if mood == track_mood else 0.0)
        
        # Artist fans normalized
        fans = getattr(track, 'artist_fans', 50.0)
        features.append(fans / 100.0)
        
        return np.array(features, dtype=np.float64)
    
    def _get_track_features(self, track_id):
        """Get cached or compute features for a track."""
        if track_id not in self._track_features_cache:
            track = self._get_track_info(track_id)
            self._track_features_cache[track_id] = self._extract_features(track)
        return self._track_features_cache[track_id]
    
    def _get_user_params(self, user):
        """Get or initialize LinUCB parameters for a user."""
        if user not in self.A:
            self.A[user] = np.eye(self.d)
            self.b[user] = np.zeros(self.d)
        return self.A[user], self.b[user]
    
    def _load_user_history(self, user):
        """Load user's listening history from Redis."""
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history
    
    def _build_user_context(self, history):
        """Build context vector from user's listening history."""
        if not history:
            return np.zeros(self.d)
        
        weighted_sum = np.zeros(self.d)
        total_weight = 0.0
        
        for track_id, listen_time in history:
            features = self._get_track_features(track_id)
            weight = min(listen_time, 2.0)  # cap at 2 seconds
            weighted_sum += weight * features
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return np.zeros(self.d)
    
    def _get_candidate_tracks(self, history, seen_tracks, max_candidates=200):
        """Get candidate tracks for recommendation."""
        candidates = []
        
        if not history:
            # Cold start: return some popular tracks
            all_track_ids = [t.track for t in self.catalog.tracks 
                           if t.track not in seen_tracks]
            return all_track_ids[:max_candidates]
        
        # Extract preferences from history
        liked_artists = defaultdict(float)
        liked_genres = defaultdict(float)
        
        for track_id, listen_time in history:
            track = self._get_track_info(track_id)
            if track is None:
                continue
            weight = min(listen_time, 2.0)
            liked_artists[track.artist] += weight
            if hasattr(track, 'genres'):
                for genre in track.genres:
                    liked_genres[genre] += weight
        
        # Score candidate tracks
        scored = []
        for track in self.catalog.tracks:
            if track.track in seen_tracks:
                continue
            
            score = 0.0
            if hasattr(track, 'artist') and track.artist in liked_artists:
                score += liked_artists[track.artist] * 3
            if hasattr(track, 'genres'):
                for genre in track.genres:
                    if genre in liked_genres:
                        score += liked_genres[genre]
            
            scored.append((track.track, score))
        
        # Sort by score and take top candidates + some random
        scored.sort(key=lambda x: x[1], reverse=True)
        top_n = int(max_candidates * 0.7)
        
        candidates = [t[0] for t in scored[:top_n]]
        
        # Add some random exploration candidates
        remaining = [t[0] for t in scored[top_n:]]
        if remaining:
            np.random.shuffle(remaining)
            candidates.extend(remaining[:max_candidates - len(candidates)])
        
        # If not enough candidates, add random tracks
        if len(candidates) < max_candidates:
            all_tracks = [t.track for t in self.catalog.tracks 
                         if t.track not in seen_tracks and t.track not in candidates]
            np.random.shuffle(all_tracks)
            candidates.extend(all_tracks[:max_candidates - len(candidates)])
        
        return candidates
    
    def _load_pretrained(self):
        """Загружает предобученные параметры LinUCB из файла."""
        import pickle
        import os
        import numpy as np
    
        pretrained_path = "./data/pretrained_bandit.pkl"
        if os.path.exists(pretrained_path):
            try:
                with open(pretrained_path, "rb") as f:
                    data = pickle.load(f)
            
                for user_str, a_matrix in data["A"].items():
                    user = int(user_str)
                    self.A[user] = np.array(a_matrix)
            
                for user_str, b_vector in data["b"].items():
                    user = int(user_str)
                    self.b[user] = np.array(b_vector)
            
                print(f"Loaded pretrained model for {len(self.A)} users")
            except Exception as e:
                print(f"Could not load pretrained model: {e}")

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        """Recommend next track using LinUCB algorithm."""
        
        # Update model with previous track feedback
        self._update_model(user, prev_track, prev_track_time)
        
        # Load history
        history = self._load_user_history(user)
        seen_tracks = set(track for track, _ in history)
        
        # Get LinUCB parameters
        A, b = self._get_user_params(user)
        
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            A_inv = np.linalg.pinv(A)
        
        theta = A_inv @ b
        
        # Get candidates
        candidates = self._get_candidate_tracks(history, seen_tracks)
        
        if not candidates:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        
        # LinUCB selection
        best_track = None
        best_score = float('-inf')
        
        for track_id in candidates:
            x = self._get_track_features(track_id)
            
            # UCB score: expected reward + exploration bonus
            expected_reward = theta @ x
            exploration_bonus = self.alpha * np.sqrt(x @ A_inv @ x)
            ucb_score = expected_reward + exploration_bonus
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_track = track_id
        
        if best_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        
        return int(best_track)
    
    def _update_model(self, user, track_id, listen_time):
        """Update LinUCB parameters based on user feedback (reward)."""
        
        x = self._get_track_features(track_id)
        
        # Reward: listen time capped at 1.0 (normalized)
        reward = min(listen_time, 1.0)
        
        A, b = self._get_user_params(user)
        
        # Sherman-Morrison update: A = A + x*x^T, b = b + reward*x
        self.A[user] = A + np.outer(x, x)
        self.b[user] = b + reward * x
