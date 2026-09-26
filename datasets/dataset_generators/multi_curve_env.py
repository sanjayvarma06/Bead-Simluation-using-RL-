import random
from pathlib import Path

import gymnasium as gym

from bead_gym_env import BeadTraceEnv


class MultiCurveEnv(gym.Env):

    def __init__(self, image_dir, seed=42):

        self.image_paths = [
            str(p)
            for p in Path(image_dir).glob("*.png")
        ]

        if not self.image_paths:
            raise ValueError(
                f"No PNG images found in {image_dir}"
            )

        self.seed_value = seed
        self.rng = random.Random(seed)

        self.current_image = None
        self.env = None

        # Create one temporary environment so SB3
        # can obtain observation/action spaces.
        self.env = BeadTraceEnv(
            image_path=self.image_paths[0],
            seed=seed
        )

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, *, seed=None, options=None):

        if seed is not None:
            self.rng.seed(seed)

        # Randomly select a curve for EVERY episode.
        self.current_image = self.rng.choice(
            self.image_paths
        )

        if self.env is not None:
            self.env.close()

        self.env = BeadTraceEnv(
            image_path=self.current_image,
            seed=self.rng.randint(0, 10_000_000)
        )

        return self.env.reset()

    def step(self, action):

        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):

        if self.env is not None:
            self.env.close()