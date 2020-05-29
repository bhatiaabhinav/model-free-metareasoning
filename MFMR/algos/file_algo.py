import gym
import numpy as np
from MFMR.monitors import utils
import random
import time as tm
from MFMR.async_algo import AsyncAlgo


class FileAlgo(AsyncAlgo):
    QUALITY_CLASS_COUNT = 100
    TIME_CLASS_COUNT = 100

    def __init__(self, problem_file_path, increment, discretization):
        super().__init__()
        self.dataset = utils.get_dataset(problem_file_path, increment)
        self.instance_id = 0
        self.state_id = 0
        self.discretization = discretization
        self._interrupted = False

    def reset(self):
        self.instance_id = random.randint(0, len(self.dataset) - 1)
        self.state_id = 0
        self._interrupted = False

    def run(self):
        '''
        Calling this method will synchronously start the algorithm. Make sure you called reset before run.
        After the method returns, `stopped` should return True.
        It is expected that this method will be invoked asynchronously using python threads or process module. The method will keep updating the appropriate variables of the algo object so that other methods like `get_obs` and `update_hyperparams` work as expected'''
        while not self._interrupted and self.state_id < len(self.dataset[self.instance_id]) - 1:
            self.state_id += 1
            # print("New iteration!", self.state_id, "q,t:", self.dataset[self.instance_id][self.state_id])
            tm.sleep(0.01)

    def update_hyperparams(self, hyperparams):
        '''sets new hyperparams. Race conditions, locks etc are taken care of here'''
        pass

    def get_hyperparam_space(self) -> gym.Space:
        ''' Hyperparameter space. Does NOT include 'STOP/CONT' space'''
        return gym.spaces.Box(np.array([]), np.array([]))

    def interrupt(self):
        '''Issues a shutdown signal to gracefully stop `run`.
        '''
        self._interrupted = True

    def get_obs_space(self) -> gym.Space:
        '''Space of `get_obs`'''
        if self.discretization:
            return gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.QUALITY_CLASS_COUNT, self.TIME_CLASS_COUNT]), shape=(2, ), dtype=np.int)
        else:
            return gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.QUALITY_CLASS_COUNT, self.TIME_CLASS_COUNT]), shape=(2, ), dtype=np.float)

    def get_obs(self):
        '''
        Returns how the algo wants to be 'seen'. Can include quality, time, and other variables - appropriately discretized sometimes'''
        raw_state = self.dataset[self.instance_id][self.state_id]
        return self.get_discretized_state(raw_state)

    def get_discretized_state(self, raw_state):
        if self.discretization:
            raw_quality, raw_time = raw_state
            quality_bounds = np.linspace(0, 1, self.QUALITY_CLASS_COUNT)
            time_bounds = np.linspace(0, self.TIME_CLASS_COUNT, self.TIME_CLASS_COUNT)
            return utils.digitize(raw_quality, quality_bounds), utils.digitize(raw_time, time_bounds)

        return raw_state

    def get_solution_quality(self):
        '''should return a float scalar - the solution quality right now.
        Need not be same as the quality exposed in get_obs. For example, quality in get_obs could be discretized
        This quality will be used to evaluate the meta-level utility'''
        raw_state = self.dataset[self.instance_id][self.state_id]
        quality, time = self.get_discretized_state(raw_state)
        return quality

    def get_time(self):
        '''should return a float scalar - some notion of time since the algorithm started. Deliberately delegated to this algo class: the algo may choose it to be simply time in seconds, or something like number of nodes expanded
        This time will be used to evaluate the meta-level utility'''
        raw_state = self.dataset[self.instance_id][self.state_id]
        quality, time = self.get_discretized_state(raw_state)
        return time

    def render(self, mode='human'):
        pass

    def close(self):
        pass
