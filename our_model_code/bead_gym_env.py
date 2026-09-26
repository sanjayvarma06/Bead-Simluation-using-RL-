"""Continuous Gymnasium wrapper for BeadEnvironment."""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from environment import BeadEnvironment

OBS_DIM = 16

class BeadTraceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, image_path: str | list | tuple | None = None, seed: int = 42):
        super().__init__()
        self._env = BeadEnvironment(image_path=image_path, seed=seed)
        self._image_path = image_path
        self._seed = seed
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        obs, info = self._env.reset()
        return np.clip(obs, -1.0, 1.0).astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return np.clip(obs, -1.0, 1.0).astype(np.float32), float(reward), terminated, truncated, info

    def render(self): pass
    def close(self): pass

def make_env(image_path: str | list | tuple | None = None, seed: int = 42):
    def _init():
        return BeadTraceEnv(image_path=image_path, seed=seed)
    return _init
