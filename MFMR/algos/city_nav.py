import logging

import numpy as np
from MFMR.algos.search_problem import SearchProblem

logger = logging.getLogger(__name__)
# ! When using the logger
# ! logs which are written for O(1) times per a search problem instance are ok to be called with info level. e.g. logger.info('Created search problem with so and so params')
# ! Any granular logs, e.g. per step logs should be called at debug level.

ldebug = logger.isEnabledFor(logging.DEBUG)
# ! ldebug is true when deepRL is invoked from command line with --debug flag.
# ! Since there might be a cost to computing the string argument (e.g. logger.debug(f'I am an expensive argument. {n}! = {fractorial(n)}'),) even when log level is above debug,
# ! it is preffered that the logger.debug(...) call be made like `ldebug and logger.debug(...)`


class CityNavigation(SearchProblem):
    '''TODO: Description of the problem goes here'''

    def __init__(self, example_arg1, example_kwarg1=None):
        # TODO: Feel free to add any args or kwargs. Can also do `*args, **kwargs`. These will be passed from MFMR.monitors.__init__.py
        # TODO: After this, fill out line 170 onwards in MFMR.monitors.__init__.py
        super().__init__()  # ! do not remove call to super class. See search_problem.py
        self.cache = {}  # To cache heuristic value of states
        # TODO: any other relevant initialization

    def seed(self, seed):
        """Set seeds of any random variables here.
        A random variable `self.random` is already provided by the super class, which is an instance of `numpy.random.RandomState`.
        It is preferred that any random stuff be done like `self.random.blablabla`. Its API is same as `numpy.random.blablabla`.
        """
        super().seed(seed)  # ! Do not remove this line. It seeds `self.random`

    def get_obs(self):
        '''returns problem specific arguments in a python list.
        ! Each argument is normalized to be between 0 a 1..
        '''
        return []

    def reset(self):
        '''Reset the search problem i.e. create a fresh instance.'''
        self.cache.clear()  # Try writing heuristic function is such a way that no need to do this.
        # TODO: Initialize code goes here.
        # TODO: After initialization. Populate self.info dictionary:
        self.info = {
            'blablabla': 'abc'
        }
        # TODO: Log some interesting info about this particular instance:
        logger.info('This instance has so and so parameters')
        raise NotImplementedError()

    def get_children_nodes(self, parent_node):
        '''Gets children "Nodes" from parent "node". `Node` class is described in search_problem.py.
        '''
        # ! It is preffered not to do anything here. `self.successors(state)` method should be implemented instead, which returns successor "states".
        # ! The superclass handles everything there is to do with "nodes".
        # ! Leave this line as such if implementing self.successors:
        return super().get_children_nodes(parent_node)

    def goal_test(self, state):
        raise NotImplementedError()

    def successors(self, state):
        """Returns an iterable (python list, iterator or generator etc.) for successor states."""
        # ! For good performance, especially for low memory consumption,
        # ! it is preferred to use `yield` style generator.
        # ! An example from npuzzle has been commented out below:
        # for action in ACTIONS:
        #     location = self.get_blank_location(state)
        #     next_location = self.get_next_blank_location(location, action)

        #     if self.is_valid_blank_location(state, next_location):
        #         next_puzzle = self.get_next_puzzle(state, action)
        #         successor = {'state': next_puzzle, 'action': action}
        #         yield successor
        raise NotImplementedError()

    def cost(self, state, action, next_state):
        raise NotImplementedError()

    def heuristic(self, state):
        raise NotImplementedError()

    def hash_state(self, state):
        '''provides a hash value which uniquely identifies the states'''
        raise NotImplementedError()
