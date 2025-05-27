from gymnasium import spaces
from gymnasium.utils import EzPickle
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

class PSO(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human"],
        "name": "pso",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self, n_pistons=20, render_mode: str | None = None):
        EzPickle.__init__(self, n_pistons=n_pistons, render_mode=render_mode)
        self.variable_ranges = [(0.01, 1, 0.01), (0.01, 1, 0.01), (0.01, 1, 0.01)]
        self.low = np.array([var[0] for var in self.variable_ranges], dtype=np.float32)
        self.high = np.array([var[1] for var in self.variable_ranges], dtype=np.float32)
        self.step_sizes = np.array([var[2] for var in self.variable_ranges], dtype=np.float32)

        self.n_pistons = n_pistons
        self.agents = ["piston_" + str(r) for r in range(self.n_pistons)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_pistons))))
        self.action_spaces = dict(
            zip(
                self.agents,
                [
                    spaces.Box(low=0, high=1, shape=(len(self.variable_ranges), 2), dtype=np.float32)
                ] * self.n_pistons,
            )
        )
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    spaces.Box(low=-np.inf, high=np.inf, shape=(5, 3), dtype=np.float32)
                ] * self.n_pistons,
            )
        )
        # self.rewards = {i: 0 for i in self.agents}
        # self.velocities = {i: np.zeros(5, dtype=np.float32) for i in self.agents}
        # self.best_positions = {i: np.zeros(5, dtype=np.float32) for i in self.agents}
        # self.best_scores = {i: float('inf') for i in self.agents}
        # self.global_best_score = float('inf')
        # self.globe_best_positions = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        # self._agent_selector = agent_selector(self.agents)
        # self.agent_selection = self._agent_selector.reset()
        # self.df = pd.read_excel("��.xlsx")
        # self.maxepoch = 100



    def reset(self, seed=None, options=None):
        pass

    def step(self, action):
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def _apply_constraints(self, list):
        pass

    def _evaluate(self, position: np.ndarray) -> float:
        """ Evaluate fitness based on the modified logic. """
        # Placeholder for user-defined evaluation logic
        position = np.array(position, dtype=np.float32)
        # Example evaluation logic (to be modified through GUI)
        # return np.sum(position)  # Simple sum of positions as a placeholder
        return position*2