import gym
import numpy as np
from MFMR.monitors import utils
import random
import time as tm
from MFMR.async_algo import AsyncAlgo


class FileAlgo(AsyncAlgo):
    QUALITY_CLASS_COUNT = 100
    TIME_CLASS_COUNT = 100

    def __init__(self, mem, problem_file_path, increment, discretization):
        super().__init__(mem)
        self.dataset = utils.get_dataset(problem_file_path, increment)
        self.instance_id = 0
        self.discretization = discretization
        self._interrupted = False

    def reset(self):
        self.instance_id = random.randint(0, len(self.dataset) - 1)
        print(f"This instance {self.instance_id} has {len(self.dataset[self.instance_id])} states")
        self.mem['state_id'] = 0
        self.mem['interrupted'] = 0

    def run(self):
        print(f"Started instance {self.instance_id}")
        while not self.mem['interrupted'] and self.mem['state_id'] < len(self.dataset[self.instance_id]) - 1:
            tm.sleep(0.25)  # Do some work
            self.mem['state_id'] += 1
            print("Did new iteration!", self.mem['state_id'], "q,t:", self.dataset[self.instance_id][self.mem['state_id']])
        print("Finished")

    def update_hyperparams(self, hyperparams):
        pass

    def get_hyperparam_space(self) -> gym.Space:
        return gym.spaces.Box(np.array([]), np.array([]))

    def interrupt(self):
        self.mem['interrupted'] = True

    def get_obs_space(self) -> gym.Space:
        if self.discretization:
            return gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.QUALITY_CLASS_COUNT, self.TIME_CLASS_COUNT]), shape=(2, ), dtype=np.int)
        else:
            return gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.QUALITY_CLASS_COUNT, self.TIME_CLASS_COUNT]), shape=(2, ), dtype=np.float)

    def get_obs(self):
        raw_state = self.dataset[self.instance_id][self.mem['state_id']]
        return self.get_discretized_state(raw_state)

    def get_discretized_state(self, raw_state):
        if self.discretization:
            raw_quality, raw_time = raw_state
            quality_bounds = np.linspace(0, 1, self.QUALITY_CLASS_COUNT)
            time_bounds = np.linspace(0, self.TIME_CLASS_COUNT, self.TIME_CLASS_COUNT)
            return utils.digitize(raw_quality, quality_bounds), utils.digitize(raw_time, time_bounds)

        return raw_state

    def get_solution_quality(self):
        raw_state = self.dataset[self.instance_id][self.mem['state_id']]
        quality, time = self.get_discretized_state(raw_state)
        return quality

    def get_time(self):
        raw_state = self.dataset[self.instance_id][self.mem['state_id']]
        quality, time = self.get_discretized_state(raw_state)
        return time

    def render(self, mode='human'):
        pass

    def close(self):
        pass
