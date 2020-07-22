import signal
import time
from multiprocessing import Process

import gym
import numpy as np  # noqa

from MFMR.async_algo import AsyncAlgo


class AsyncAlgoMonitor(gym.Env):
    STOP_ACTION = 0
    CONTINUE_ACTION = 1

    def __init__(self, alpha, beta, monitoring_interval, algo_cls, *algo_args, **algo_kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.monitoring_interval = monitoring_interval
        self.algo = algo_cls(*algo_args, **algo_kwargs)  # type: AsyncAlgo
        self.run_process = None

        # set observation space.
        self.observation_space = self.algo.get_obs_space()

        # set action space
        # It is a mixed action space: Dicrete(2) X HyperparamSpace
        self.hyperparam_space = self.algo.get_hyperparam_space()
        self.has_hyperparams = self.hyperparam_space.shape[0] > 0
        if self.has_hyperparams:
            self.action_space = gym.spaces.Tuple(
                (gym.spaces.Discrete(2), self.hyperparam_space))
        else:
            self.action_space = gym.spaces.Discrete(2)  # Just STOP/CONTINUE

    def reset(self):
        if self.run_process is not None:
            self.run_process.terminate()
            self.run_process = None

        assert self.run_process is None, "Old Episode Still Exists?"
        self.algo.reset()
        obs = self.algo.get_obs()
        self.cur_utility = self.get_cur_utility()
        self.run_process = Process(target=self.algo.run)
        self.run_process.start()
        return obs

    def step(self, action):
        '''Gym API. Returns observation, reward, done, info_dict.
        The action space is 2 x H, where H is the hyperparam space of the object level algo.
        Example actions: (0, np.array([...])) or (1, np.array([...])).
        0 = STOP
        1 = CONTINUE
        '''
        assert self.action_space.contains(
            action), f"Invalid action. Action space is {self.action_space}"
        if self.has_hyperparams:
            cont_decision, hyperparams = action
        else:
            cont_decision = action

        done = False
        info = {'interrupted': False, 'graceful_exit': True}
        self.prev_utility = self.cur_utility

        if not self.run_process.is_alive():
            # automatically completed ... found the "final" solution.
            self.run_process.join()
            done = True
            info['interrupted'] = False
            info['graceful_exit'] = True
        else:
            if not cont_decision:
                self.algo.interrupt()
                # give it a second to allow graceful termination
                self.run_process.join(timeout=1)
                if self.run_process.exitcode is None:  # i.e. not yet terminated
                    print('Sending SIGTERM to process')
                    while self.run_process.exitcode is not None:
                        self.run_process.terminate()
                        self.run_process.join(timeout=0.5)
                done = True
                info['interrupted'] = True
                info['graceful_exit'] = not (
                    self.run_process.exitcode == -signal.SIGTERM)
            else:
                if self.has_hyperparams:
                    self.algo.update_hyperparams(hyperparams)
                time.sleep(self.monitoring_interval)

        info['solution_quality'] = self.algo.get_solution_quality()
        info['time'] = self.algo.get_time()
        self.cur_utility = self.get_cur_utility()
        info['utility'] = self.cur_utility

        reward = self.cur_utility - self.prev_utility

        return self.algo.get_obs(), reward, done, info

    def get_solution_quality(self):
        return self.algo.get_solution_quality()

    def get_time(self):
        return self.algo.get_time()

    def get_cur_utility(self):
        '''time dependent utility'''
        quality = self.algo.get_solution_quality()
        t = self.algo.get_time()
        return self.alpha * quality - np.exp(self.beta * t)

    def seed(self, seed):
        super().seed()
        self.algo.seed(seed)

    def render(self, mode='human'):
        '''GYM API. Some Nice Visualization'''
        return self.algo.render(mode=mode)

    def close(self):
        '''GYM API. Close and cleanup any rendering related objects'''
        if self.run_process is not None:
            self.run_process.terminate()
            self.run_process = None
        return self.algo.close()
