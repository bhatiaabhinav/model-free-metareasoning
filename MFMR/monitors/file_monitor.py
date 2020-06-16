import random

import gym
import numpy as np

from . import utils


class FileBasedAlgo(gym.Env):
    STOP_ACTION = 0
    CONTINUE_ACTION = 1

    QUALITY_CLASS_COUNT = 100
    TIME_CLASS_COUNT = 100

    def __init__(self, problem_file_path, alpha, beta, increment, discretization):
        super().__init__()

        self.dataset = utils.get_dataset(problem_file_path, increment)

        self.instance_id = 0
        self.state_id = 0

        self.alpha = alpha
        self.beta = beta

        self.discretization = discretization

        if self.discretization:
            self.observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array(
                [self.QUALITY_CLASS_COUNT, self.TIME_CLASS_COUNT]), shape=(2, ), dtype=np.int)
        else:
            self.observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array(
                [self.QUALITY_CLASS_COUNT, self.TIME_CLASS_COUNT]), shape=(2, ), dtype=np.float)
        self.action_space = gym.spaces.Discrete(2)

    def get_reward(self):
        previous_quality, previous_time = self.get_previous_state()
        current_quality, current_time = self.get_current_state()

        previous_utility = utils.get_time_dependent_utility(
            previous_quality, previous_time, self.alpha, self.beta)
        current_utility = utils.get_time_dependent_utility(
            current_quality, current_time, self.alpha, self.beta)

        return current_utility - previous_utility

    def get_utility(self):
        raw_state = self.dataset[self.instance_id][self.state_id]
        quality, time = self.get_discretized_state(raw_state)
        return utils.get_time_dependent_utility(quality, time, self.alpha, self.beta)

    def get_optimal_utility(self):
        max_utility = float("-inf")

        for raw_state in self.dataset[self.instance_id]:
            quality, time = self.get_discretized_state(raw_state)
            utility = utils.get_time_dependent_utility(
                quality, time, self.alpha, self.beta)

            if utility > max_utility:
                max_utility = utility

        return max_utility

    def get_discretized_state(self, raw_state):
        if self.discretization:
            raw_quality, raw_time = raw_state
            quality_bounds = np.linspace(0, 1, self.QUALITY_CLASS_COUNT)
            time_bounds = np.linspace(
                0, self.TIME_CLASS_COUNT, self.TIME_CLASS_COUNT)
            return utils.digitize(raw_quality, quality_bounds), utils.digitize(raw_time, time_bounds)

        return raw_state

    def get_previous_state(self):
        raw_state = self.dataset[self.instance_id][self.state_id - 1]
        return self.get_discretized_state(raw_state)

    def get_current_state(self):
        raw_state = self.dataset[self.instance_id][self.state_id]
        return self.get_discretized_state(raw_state)

    def is_last_instance(self):
        return self.instance_id == len(self.dataset) - 1

    def is_episode_done(self):
        return self.state_id >= len(self.dataset[self.instance_id]) - 1

    def reset(self):
        self.instance_id = random.randint(0, len(self.dataset) - 1)
        self.state_id = 0
        return self.get_current_state()

    def step(self, action):
        if action == self.STOP_ACTION or self.is_episode_done():
            return self.get_current_state(), 0, True, {}

        self.state_id += 1
        return self.get_current_state(), self.get_reward(), False, {}

    def render(self, mode='human'):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()
