from .anytime_algo import AnytimeAlgo


class RRTStar(AnytimeAlgo):
    def __init__(self, gamma=0.99, map_size=(1, 1), random_map_every_reset=True, n_obstacles=2):
        super().__init__(self, gamma=gamma)
        self.random_map_every_reset = random_map_every_reset
        self.map_size = map_size
        self.n_obstacles = n_obstacles
        # set observation and action space here. Refer to template in anytime_algo.py

    def reset(self):
        # Create a new map here
        ####

        ####
        return super().reset()

    def step(self, action):
        return super().step(action)

    def get_observation(self):
        '''return solution_quality, time etc and optionally more summary stats of best path so far
        e.g. return np.asarray([self.get_solution_utility(), self.t])
        '''
        raise NotImplementedError()

    def get_done(self):
        '''whether anytime algo converged'''
        raise NotImplementedError()

    def update_hyperparams(self, action):
        '''For example set self.should_stop=True if action is STOP
        Or set things like new branching factor'''
        raise NotImplementedError()

    def improve_solution(self):
        '''continue execution of the algo with updated hyperparams till next best path is found. Keep track of self.time'''
        raise NotImplementedError()

    def get_solution_utility(self):
        '''overall utility of the solution so far. Includes time cost'''
        raise NotImplementedError()

    def render(self, mode='human'):
        '''GYM API. Some Nice Visualization'''
        raise NotImplementedError()

    def close(self):
        '''GYM API. Close and cleanup any rendering related objects'''
        pass
