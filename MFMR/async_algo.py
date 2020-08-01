from multiprocessing import Manager
from typing import List

import gym
import numpy as np


class AsyncAlgo:
    '''Base class for implemented asynchronously monitored anytime algorithms.'''

    def __init__(self):
        '''
        The concrete class can set default hyperparameters and settings here using constructor arguments.
        mem should be a shared memory dictionary created using process.Manager().dict()
        '''
        self.mem = Manager().dict()
        self.mem['action'] = 0
        self.random = np.random.RandomState()

    def reset(self):
        '''
        Reset the state of the algo to default. Default need not mean a deterministic default. It could be sampled from a set of start states.
        This method will be called before or after a `run` and is not expected to be called during a run.
        '''
        raise NotImplementedError()

    def run(self):
        '''
        Make sure you called reset before run. It is expected that this method will be invoked as a seperate process using process module. The method will keep updating the appropriate variables in self.mem so that other methods like `get_obs` and `update_hyperparams` work as expected'''
        raise NotImplementedError()

    def interrupt(self):
        '''Issues a shutdown signal to gracefully stop `run`.
        '''
        raise NotImplementedError()

    def get_obs_space(self) -> gym.Space:
        '''Space of `get_obs`'''
        raise NotImplementedError()

    def get_obs(self):
        '''
        Returns how the algo wants to be 'seen'. Can include quality, time, and other variables - appropriately discretized sometimes'''
        raise NotImplementedError()

    def get_info(self):
        '''A dictionary containing any info of interest. Useful for analysis and logging'''
        return {}

    def get_action_space(self) -> gym.Space:
        '''space of actions (excluding STOP/CONTINUE). If dicrete, should be atleast Dicrete(1), with action 0 being NOOP.
        TODO: Only Dicrete action spaces are supported right now'''
        return gym.spaces.Discrete(1)

    def get_action_meanings(self) -> List[str]:
        '''a list of action meaning strings correspoding to each action's integer code. First item in the list should always be "NOOP"'''
        return ['NOOP']

    def set_action(self, action):
        '''sets self.mem['action'] = action. This action does not include STOP/CONTINUE. 0 always means NOOP. Other action codes should be used to update hyperparams'''
        assert self.get_action_space().contains(action)
        self.mem['action'] = action

    def get_solution_quality(self):
        '''should return a float scalar - the solution quality right now.
        Need not be same as the quality exposed in get_obs. For example, quality in get_obs could be discretized
        This quality will be used to evaluate the meta-level utility'''
        raise NotImplementedError()

    def get_time(self):
        '''should return a float scalar - some notion of time since the algorithm started. Deliberately delegated to this algo class: the algo may choose it to be simply time in seconds, or something like number of nodes expanded.
        This time could be different from the time exposed in get_obs. For eg. time in get_obs could be discretized.
        This time will be used to evaluate the meta-level utility
        '''
        raise NotImplementedError()

    def seed(self, seed):
        self.random.seed(seed)

    def render(self, mode='human'):
        pass

    def close(self):
        self.mem.clear()
