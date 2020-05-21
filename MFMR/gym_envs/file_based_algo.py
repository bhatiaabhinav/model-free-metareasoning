import numpy as np

import gym


class FileBasedAlgo(gym.Env):
    def __init__(self, gamma=0.99):
        super().__init__()
        self.gamma = gamma

    def reset(self):
        self.time = 0
        return self.get_observation()

    def step(self, action):
        return None, None, None, None

    def improve_solution(self):
        raise NotImplementedError()

    def get_observation(self):
        raise NotImplementedError()

    def get_solution_utility(self):
        raise NotImplementedError()

    def get_done(self):
        raise NotImplementedError()

    def render(self, mode='human'):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()
