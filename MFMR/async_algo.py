import gym
import numpy as np


class AsyncAlgo:
    '''Base class for implemented asynchronously monitored anytime algorithms.'''

    def __init__(self, mem):
        '''
        The concrete class can set default hyperparameters and settings here using constructor arguments.
        mem should be a shared memory dictionary created using process.Manager().dict()
        '''
        self.mem = mem

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

    def update_hyperparams(self, hyperparams):
        '''sets new hyperparams. Race conditions, locks etc are taken care of here'''
        pass

    def get_hyperparam_space(self) -> gym.Space:
        ''' Hyperparameter space. Does NOT include 'STOP/CONT' space'''
        return gym.spaces.Box(np.array([]), np.array([]))

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

    def render(self, mode='human'):
        pass

    def close(self):
        pass
