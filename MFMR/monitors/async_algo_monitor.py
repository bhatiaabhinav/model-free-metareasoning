from multiprocessing import Process

import gym
import numpy as np  # noqa

from MFMR.async_algo import AsyncAlgo


class AsyncAlgoMonitor(gym.Env):
    STOP_ACTION = 0
    CONTINUE_ACTION = 1

    def __init__(self, algo: AsyncAlgo, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.algo = algo
        self.run_process = None

        # set observation space.
        self.observation_space = self.algo.get_obs_space()

        # set action space
        # It is a mixed action space: Dicrete(2) X HyperparamSpace
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(2), self.algo.get_hyperparam_space()))

    def reset(self):
        if self.run_process is not None:
            self.run_process.terminate()
            self.run_process = None

        assert self.run_process is None, "Old Episode Still Exists?"
        self.algo.reset()
        obs = self.algo.get_obs()
        self.run_process = Process(target=self.algo.run)
        self.run_process.start()
        self.cur_utility = self.get_cur_utility()
        return obs

    def step(self, action):
        '''Gym API. Returns observation, reward, done, info_dict.
        The action space is 2 x H, where H is the hyperparam space of the object level algo.
        Example actions: (0, np.array([...])) or (1, np.array([...])).
        0 = STOP
        1 = CONTINUE
        '''
        assert self.action_space.contains(action), f"Invalid action. Action space is {self.action_space}"
        cont_decision, hyperparams = action

        done = False
        info = {'interrupted': False}
        self.prev_utility = self.cur_utility

        if not self.run_process.is_alive():
            # automatically completed ... found the "final" solution.
            self.run_process.join()
            done = True
            info['interrupted'] = False
        else:
            if not cont_decision:
                self.algo.interrupt()
                try:
                    self.run_process.join(timeout=1)
                except e:
                    self.run_process.terminate()
                done = True
                info['Interrupted'] = True
            else:
                self.algo.update_hyperparams(hyperparams)

        info['solution_quality'] = self.algo.get_solution_quality()
        info['time'] = self.algo.get_time()
        self.cur_utility = self.get_cur_utility()
        info['utility'] = self.cur_utility

        reward = self.cur_utility - self.prev_utility

        return self.algo.get_obs(), reward, done, info

    def get_cur_utility(self):
        '''time dependent utility'''
        quality = self.algo.get_solution_quality()
        time = self.algo.get_time()
        return self.alpha * quality - np.exp(self.beta * time)

    def render(self, mode='human'):
        '''GYM API. Some Nice Visualization'''
        return self.algo.render(mode=mode)

    def close(self):
        '''GYM API. Close and cleanup any rendering related objects'''
        return self.algo.close()


def main():
    print("Testing the environment...")
    from MFMR.algos.file_algo import FileAlgo
    algo = FileAlgo("problems/test.json", 5, True)
    env = AsyncAlgoMonitor(algo, 200, 0.3)
    import time as tm
    import random

    random.seed(0)

    hyperparams = np.array([])

    print("Running episode 1...")
    print(env.reset())
    tm.sleep(1)
    print(env.step((env.CONTINUE_ACTION, hyperparams)))
    tm.sleep(1)
    print(env.step((env.CONTINUE_ACTION, hyperparams)))
    tm.sleep(1)
    print(env.step((env.CONTINUE_ACTION, hyperparams)))
    tm.sleep(1)
    print(env.step((env.STOP_ACTION, hyperparams)))

    print("Running episode 2...")
    print(env.reset())
    tm.sleep(0.01)
    print(env.step((env.CONTINUE_ACTION, hyperparams)))
    tm.sleep(0.01)
    print(env.step((env.CONTINUE_ACTION, hyperparams)))
    tm.sleep(0.01)
    print(env.step((env.STOP_ACTION, hyperparams)))
    tm.sleep(0.01)

    print("Running episode 3...")
    print(env.reset())
    print(env.step((env.CONTINUE_ACTION, hyperparams)))
    print(env.step((env.CONTINUE_ACTION, hyperparams)))
    print(env.step((env.CONTINUE_ACTION, hyperparams)))
    print(env.step((env.CONTINUE_ACTION, hyperparams)))
    print(env.step((env.STOP_ACTION, hyperparams)))


if __name__ == "__main__":
    main()
