import gym
import numpy as np  # noqa


class AnytimeAlgo(gym.Env):
    def __init__(self, gamma=0.99):
        super().__init__()
        self.gamma = gamma  # time cost gamma

        # set observation space.
        # for example if it is quality x time:
        # self.observation_space = gym.spaces.Box(low=np.asarray(
        #     [-np.inf, 0]), high=np.asarray([np.inf, np.inf]), shape=(2,), dtype=np.float32)

        # set action space
        # for example if it is one of STOP or CONT. i.e. 2 discrete options:
        # self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        '''Gym API. Returns observation'''
        self.time = 0
        return self.get_observation()

    def step(self, action):
        '''Gym API. Returns observation, reward, done, info_dict
        '''
        # 1. Decide to continue or stop algo based on action
        # 2. Set new hyperparams based on action
        # .......... #
        self.update_hyperparams(action)
        # .......... #

        old_utility = self.get_solution_utility()

        # 3. Continue algorithm till next solution
        # .......... #
        self.improve_solution()
        # .......... #

        # 4. Get New Observation
        obs = self.get_observation()

        # 5. Compute change in utility as reward
        cur_utility = self.get_solution_utility()
        r = cur_utility - old_utility

        # 6. Has the algorithm converged?
        d = self.get_done()

        # 7. Any extra info to return for debugging purposes? e.g. split up of overall utility
        info = {
            'total_utility': cur_utility,
            'time_cost': - self.gamma ** self.time
        }

        return obs, r, d, info

    def update_hyperparams(self, action):
        '''For example set self.should_stop=True if action is STOP'''
        raise NotImplementedError()

    def improve_solution(self):
        raise NotImplementedError()

    def get_observation(self):
        '''return solution_quality, time etc
        e.g. return np.asarray([self.get_solution_utility(), self.t])
        '''
        raise NotImplementedError()

    def get_solution_utility(self):
        raise NotImplementedError()

    def get_done(self):
        '''whether anytime algo converged'''
        raise NotImplementedError()

    def render(self, mode='human'):
        '''GYM API. Some Nice Visualization'''
        raise NotImplementedError()

    def close(self):
        '''GYM API. Close and cleanup any rendering related objects'''
        pass
