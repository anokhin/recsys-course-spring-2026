import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Dict

from .config import RecEnvConfig
from .track import TrackCatalog
from .user import UserCatalog


class RecEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: RecEnvConfig, collect_negatives: int = 0):
        super(RecEnv, self).__init__()
        self.config = config
        self.collect_negatives = collect_negatives

        self.track_catalog = TrackCatalog(config.track_catalog_config)
        self.user_catalog = UserCatalog(config.user_catalog_config)

        # At each step you suggest a track, so each action is a single track ID
        self.action_space = Discrete(self.track_catalog.size())

        # We need to provide a user ID to the recommender and the initial track
        self.observation_space = Dict(
            user=Discrete(self.user_catalog.size()),
            track=Discrete(self.track_catalog.size()),
        )

        self.user = None
        self.session = None

        self.reset()

    def step(self, recommendation: int):
        assert self.action_space.contains(recommendation), str(recommendation)
        info = {}
        if self.collect_negatives > 0:
            session = self.session
            n = self.track_catalog.size()
            pool = [
                t
                for t in range(n)
                if t not in session.seen_tracks and t != recommendation
            ]
            k = min(self.collect_negatives, len(pool))
            if k > 0:
                neg = np.random.choice(pool, size=k, replace=False)
                info["negative_tracks"] = [int(x) for x in neg.tolist()]
            else:
                info["negative_tracks"] = []

        playback = self.user.consume(
            recommendation, self.session, self.track_catalog
        )
        terminated = self.session.finished
        truncated = False
        info["duplicate"] = playback.duplicate
        info["affinity"] = (
            None if playback.affinity is None else float(playback.affinity)
        )
        info["recommended_artist"] = playback.artist
        return self.session.observe(), playback.time, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.user = self.user_catalog.sample_user()
        self.session = self.user.new_session(self.track_catalog)
        return self.session.observe(), {}

    def render(self, mode="human", close=False):
        print(f"Current session: {self.session}")

    def seed(self, seed=None):
        np.random.seed(seed)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
