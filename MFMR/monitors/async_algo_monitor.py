import logging
import signal
import time
from multiprocessing import Process

import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np  # noqa
from gym.envs.classic_control.rendering import SimpleImageViewer

from MFMR.algos.aastar import logger
from MFMR.async_algo import AsyncAlgo

# mpl.rcParams['text.antialiased'] = False
fig = plt.gcf()
fig.set_size_inches(4, 3)
ax = plt.gca()
plt.setp([ax.get_xticklines() + ax.get_yticklines() +
          ax.get_xgridlines() + ax.get_ygridlines()], antialiased=False)


class AsyncAlgoMonitor(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'save_png']}
    STOP_ACTION = None
    CONTINUE_ACTION = 0

    def __init__(self, alpha, beta_options, monitoring_interval, observe_beta, algo_cls, *algo_args, **algo_kwargs):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.random = np.random.RandomState()
        self.alpha = alpha
        self.beta_options = beta_options
        self.beta = self.random.choice(self.beta_options)
        self.monitoring_interval = monitoring_interval
        self.observe_beta = observe_beta
        self.algo = algo_cls(*algo_args, **algo_kwargs)  # type: AsyncAlgo
        self.run_process = None

        # set observation space.
        if self.observe_beta:
            algo_obs_space = self.algo.get_obs_space()  # type: gym.spaces.Box
            low = np.concatenate(
                (algo_obs_space.low, np.array([0], dtype=np.float32)))
            high = np.concatenate(
                (algo_obs_space.high, np.array([1], dtype=np.float32)))
            self.observation_space = gym.spaces.Box(
                low, high, dtype=np.float32)
        else:
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

        self.viewer = None
        self.episode_id = -1

    def get_action_meanings(self):
        return self.action_meanings

    def reset(self):
        self.logger.info('Resetting env')
        self.episode_id += 1
        self.terminate_process()

        self.beta = self.random.choice(self.beta_options)
        self.algo.reset()
        self.cur_utility = self.get_cur_utility()
        self.run_process = Process(target=self.algo.run)
        self.run_process.start()
        self.render_ts = []
        self.render_qs = []
        self.render_utils = []
        self.render_ws = []
        self.render_q_ubs = []
        self.logger.info('Waiting for process alive')
        while not self.run_process.is_alive():
            time.sleep(0.001)
        self.logger.info('process now alive')
        self.last_step_at = time.time()
        return self.get_obs()

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
        logger.debug('stepping')
        assert self.action_space.contains(
            action), f"Invalid action. Action space is {self.action_space}"

        cont_decision = action < self.action_space.n - 1  # i.e. not 'STOP'

        done = False
        info = {'beta': self.beta, 'interrupted': 0, 'graceful_exit': 1}
        self.prev_utility = self.cur_utility

        if not self.run_process.is_alive():
            # automatically completed ... found the "final" solution.
            self.run_process.join()
            done = True
            self.logger.info('natural done')
            info['interrupted'] = 0
            info['graceful_exit'] = 1
        else:
            if not cont_decision:
                done = True
                self.logger.info('interrupt done')
                info['interrupted'] = 1
                info['graceful_exit'] = int(self.terminate_process())
            else:
                self.algo.set_action(action)
                overhead = time.time() - self.last_step_at
                time.sleep(max(self.monitoring_interval - overhead, 0))

        self.cur_utility = self.get_cur_utility()
        info['utility'] = self.cur_utility
        # info['image'] = self.render(mode='rgb_array')
        info.update(self.algo.get_info())

        reward = self.cur_utility - self.prev_utility
        reward = reward * 10 / self.alpha  # scale reward to make it easier to learn

        self.render_ts.append(self.get_time())
        self.render_qs.append(10 * self.get_solution_quality())
        self.render_utils.append(self.get_cur_utility() * 10 / self.alpha)
        self.render_ws.append(info['w'])
        ws = self.render_ws
        self.render_q_ubs.append(10 * info['q_ub'])
        info['w_av'] = np.mean(ws)
        info['w_std'] = np.std(ws)
        ep_len = len(ws)
        part_lenths = np.array([ep_len // 4] * 4)
        remainder = ep_len % 4
        if remainder > 0:
            part_lenths[0:remainder] = part_lenths[0:remainder] + 1
        if ep_len >= 4:
            # print(part_lenths)
            info['w1'] = np.mean(ws[0:part_lenths[0]])
            info['w2'] = np.mean(
                ws[part_lenths[0]:part_lenths[0] + part_lenths[1]])
            info['w3'] = np.mean(
                ws[part_lenths[0] + part_lenths[1]:part_lenths[0] + part_lenths[1] + part_lenths[2]])
            info['w4'] = np.mean(
                ws[part_lenths[0] + part_lenths[1] + part_lenths[2]:])

        self.last_step_at = time.time()

        if (self.episode_id + 1) % 20 == 0:
            if done:
                x = self.render_ts
                plt.clf()
                plt.plot(x, self.render_qs, label='10 * q', color='blue')
                plt.plot(x, self.render_utils,
                         label=f'10 * util/α', color='red')
                plt.plot(x, self.render_ws, label='w', color='brown')
                plt.plot(x, self.render_q_ubs,
                         label='10 * q_ub', color='green')
                plt.legend()
                plt.grid()
                # plt.tight_layout()
                fig.canvas.draw()
                data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                # from RL.__main__ import logdir
                # import os
                # path = os.path.join(
                #     logdir, 'images', f'ep{self.episode_id}.png')
                # plt.savefig(path)
                time_costs = {
                    0.3: 'None',
                    0.4: 'Low',
                    0.5: 'Medium',
                    0.6: 'High',
                    0.7: 'V High'
                }
                import wandb
                wandb.log({"Plot": wandb.Image(
                    img, caption=f'{self.episode_id + 1} Episodes (Time cost = {time_costs[self.beta]}, TSP N = {info["tsp_n"]}, Sparsity={np.round(info["tsp_sparsity"], 1)})')})
        return self.get_obs(), reward, done, info

    def get_obs(self):
        if self.observe_beta:
            return np.concatenate((np.asarray([self.beta], dtype=np.float32), self.algo.get_obs()))
        else:
            return self.algo.get_obs()

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
        self.random.seed(seed)
        self.algo.seed(seed)

    def render(self, mode='human'):
        '''GYM API. Some Nice Visualization'''
        # print('Creating visualization')
        info = self.algo.get_info()
        # self.render_infos.append(info)
        self.render_ts.append(self.get_time())
        self.render_qs.append(10 * self.get_solution_quality())
        self.render_utils.append(self.get_cur_utility() * 10 / self.alpha)
        x = self.render_ts
        plt.clf()
        plt.plot(x, self.render_qs, label='10 * q', color='blue')
        plt.plot(x, self.render_utils, label=f'10 * util/α', color='red')
        if 'w' in info:
            self.render_ws.append(info['w'])
            plt.plot(x, self.render_ws, label='w', color='brown')
        if 'q_ub' in info:
            self.render_q_ubs.append(10 * info['q_ub'])
            plt.plot(x, self.render_q_ubs, label='10 * q_ub', color='green')
        plt.legend()
        plt.grid()
        # plt.tight_layout()
        fig.canvas.draw()
        if mode == 'save_png':
            plt.savefig(f'ep{self.episode_id}.png')
            return
        data = np.fromstring(fig.canvas.tostring_rgb(),
                             dtype=np.uint8)
        img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if mode == 'rgb_array':
            # print('created')
            return img
        elif mode == 'human':
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            # X = np.array(fig.canvas.renderer.buffer_rgba())
            # print('created')
            self.viewer.imshow(img)
        else:
            raise ValueError(f'Render mode {mode} not supported')

    def close(self):
        '''GYM API. Close and cleanup any rendering related objects'''
        self.logger.info('Closing env')
        if self.viewer is not None:
            self.viewer.close()
        self.terminate_process()
        return self.algo.close()
