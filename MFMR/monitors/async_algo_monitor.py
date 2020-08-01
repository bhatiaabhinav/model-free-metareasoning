import logging
import signal
import time
from multiprocessing import Process

import gym
import numpy as np  # noqa

from MFMR.async_algo import AsyncAlgo


class AsyncAlgoMonitor(gym.Env):
    STOP_ACTION = None
    CONTINUE_ACTION = 0

    def __init__(self, alpha, beta, monitoring_interval, algo_cls, *algo_args, **algo_kwargs):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.alpha = alpha
        self.beta = beta
        self.monitoring_interval = monitoring_interval
        self.algo = algo_cls(*algo_args, **algo_kwargs)  # type: AsyncAlgo
        self.run_process = None

        # set observation space.
        self.observation_space = self.algo.get_obs_space()

        # set action space
        algo_ac_space = self.algo.get_action_space()
        assert isinstance(algo_ac_space, gym.spaces.Discrete)
        self.action_space = gym.spaces.Discrete(
            algo_ac_space.n + 1)  # add stop as the last action
        self.action_meanings = self.algo.get_action_meanings() + ['STOP']
        self.STOP_ACTION = self.action_space.n - 1
        assert self.action_meanings[0] == 'NOOP'
        assert len(self.action_meanings) == self.action_space.n

    def get_action_meanings(self):
        return self.action_meanings

    def reset(self):
        self.logger.info('Resetting env')
        self.terminate_process()

        self.algo.reset()
        obs = self.algo.get_obs()
        self.cur_utility = self.get_cur_utility()
        self.run_process = Process(target=self.algo.run)
        self.run_process.start()
        return obs

    def terminate_process(self):
        '''terminates the algo, first by invoking interrupt method and giving it a chance to terminate gracefully,
        and if that does not work within a second, then by sending SIGTERM to the algo process.
        returns whether the termination was graceful'''
        graceful = True
        if self.run_process is not None and self.run_process.is_alive():
            self.algo.interrupt()
            self.logger.info('waiting for graceful termination')
            # give it a second to allow graceful termination
            self.run_process.join(timeout=1)
            if self.run_process.exitcode is None:  # i.e. not yet terminated
                self.logger.debug('Sending SIGTERM to process')
                while self.run_process.exitcode is None:
                    self.run_process.terminate()
                    self.run_process.join(timeout=0.5)
                graceful = False
            self.logger.info(f'Graceful termination {graceful}')
        return graceful

    def step(self, action):
        '''Gym API. Returns observation, reward, done, info_dict.
        The action space is 2 x H, where H is the hyperparam space of the object level algo.
        Example actions: (0, np.array([...])) or (1, np.array([...])).
        0 = STOP
        1 = CONTINUE
        '''
        assert self.action_space.contains(
            action), f"Invalid action. Action space is {self.action_space}"

        cont_decision = action < self.action_space.n - 1  # i.e. not 'STOP'

        done = False
        info = {'interrupted': False, 'graceful_exit': True}
        self.prev_utility = self.cur_utility

        if not self.run_process.is_alive():
            # automatically completed ... found the "final" solution.
            self.run_process.join()
            done = True
            self.logger.info('natural done')
            info['interrupted'] = False
            info['graceful_exit'] = True
        else:
            if not cont_decision:
                done = True
                self.logger.info('interrupt done')
                info['interrupted'] = True
                info['graceful_exit'] = self.terminate_process()
            else:
                self.algo.set_action(action)
                time.sleep(self.monitoring_interval)

        info['solution_quality'] = self.algo.get_solution_quality()
        info['time'] = self.algo.get_time()
        self.cur_utility = self.get_cur_utility()
        info['utility'] = self.cur_utility
        info.update(self.algo.get_info())

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
        self.logger.info('Closing env')
        self.terminate_process()
        return self.algo.close()
