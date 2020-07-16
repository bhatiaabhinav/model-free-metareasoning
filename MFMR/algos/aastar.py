# import itertools
# import os
# import re
# import sys
import time as tm
from multiprocessing import Manager

import gym
import numpy as np

from MFMR import utils
from MFMR.algos.search_problem import Node, OpenList, SearchProblem, get_key
from MFMR.async_algo import AsyncAlgo

FILE_TEMPLATE = """NAME : %s
COMMENT : %s
TYPE : TSP
DIMENSION: %d
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
%s
EOF
"""
CITY_TEMPLATE = "%d %i %i %s"
COMMENT = "No Comment"
CITY_PATTERN = "\d+ (\d+) (\d+)"  # noqa
DELIMITER = "\n"

ITERATIONS = 100


class AAstar(AsyncAlgo):
    QUALITY_CLASS_COUNT = 100
    TIME_CLASS_COUNT = 100

    def __init__(self, weight, discretization, search_problem_cls, *search_problem_args, **search_problem_kwargs):
        super().__init__(Manager().dict())
        self.iterations = ITERATIONS
        self.problem = search_problem_cls(
            *search_problem_args, **search_problem_kwargs)  # type: SearchProblem
        self.weight = weight
        self.discretization = discretization

    # ------------------------------------------ helper methods -----------------------------------------------

    # ----------------------------------------------------------------------------------------------------------

    def reset(self):
        self.problem.reset()
        self.mem['cost'] = np.inf
        self.start_heuristic = self.problem.heuristic(self.problem.start_state)
        self.mem['start_time'] = tm.time()
        self.mem['time'] = 0
        self.mem['interrupted'] = False

    def run(self):
        problem = self.problem

        start_node = Node(problem.start_state)
        start_node_key = get_key(start_node.state)

        open_list = OpenList()
        open_list.add(start_node, self.weight *
                      problem.heuristic(start_node.state))

        closed_set = set()
        best_node_value = float('inf')

        path_costs = {start_node_key: start_node.path_cost}
        open_list_size = 1
        closed_set_size = 0
        self.mem['open_list_size'] = open_list_size
        self.mem['closed_set_size'] = closed_set_size

        while open_list:
            current_node = open_list.remove()
            open_list_size = open_list_size - 1
            current_node_key = get_key(current_node.state)
            current_node_value = current_node.path_cost + \
                problem.heuristic(current_node.state)
            self.mem['open_list_size'] = open_list_size
            self.mem['closed_set_size'] = closed_set_size

            if current_node_value < best_node_value:
                closed_set.add(current_node_key)
                closed_set_size = closed_set_size + 1
                self.mem['open_list_size'] = open_list_size
                self.mem['closed_set_size'] = closed_set_size

                for child_node in problem.get_children_nodes(current_node):
                    child_node_value = child_node.path_cost + \
                        problem.heuristic(child_node.state)
                    child_node_key = get_key(child_node.state)

                    if child_node_value < best_node_value:
                        if problem.goal_test(child_node.state):
                            path_costs[child_node_key] = child_node.path_cost
                            best_node_value = child_node_value

                            self.mem['solution'] = child_node.get_solution()
                            self.mem['cost'] = len(self.mem['solution'])
                        elif child_node_key in closed_set or child_node in open_list:
                            if path_costs[child_node_key] > child_node.path_cost:
                                path_costs[child_node_key] = child_node.path_cost

                                if child_node_key in closed_set:
                                    open_list.add(
                                        child_node, path_costs[child_node_key] + self.weight * problem.heuristic(child_node.state))
                                    open_list_size += 1
                                    closed_set.remove(child_node_key)
                                    closed_set_size -= 1
                                self.mem['open_list_size'] = open_list_size
                                self.mem['closed_set_size'] = closed_set_size
                        else:
                            path_costs[child_node_key] = child_node.path_cost
                            open_list.add(
                                child_node, path_costs[child_node_key] + self.weight * problem.heuristic(child_node.state))
                            open_list_size += 1
                            self.mem['open_list_size'] = open_list_size

                    self.mem['time'] = tm.time() - self.mem['start_time']
                    if self.mem['interrupted']:
                        break

            self.mem['time'] = tm.time() - self.mem['start_time']
            if self.mem['interrupted']:
                break

    def update_hyperparams(self, hyperparams):
        pass

    def get_hyperparam_space(self) -> gym.Space:
        return gym.spaces.Box(low=np.array([]), high=np.array([]))

    def interrupt(self):
        self.mem['interrupted'] = True

    def get_obs_space(self) -> gym.Space:
        if self.discretization:
            return gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.QUALITY_CLASS_COUNT, self.TIME_CLASS_COUNT]), shape=(2, ), dtype=np.int)
        else:
            return gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.QUALITY_CLASS_COUNT, self.TIME_CLASS_COUNT]), shape=(2, ), dtype=np.float)

    def get_obs(self):
        self.mem['time'] = tm.time() - self.mem['start_time']
        cost, time = self.mem['cost'], self.mem['time']
        quality = self.start_heuristic / cost
        return self.get_discretized_state((quality, time))

    def get_discretized_state(self, raw_state):
        if self.discretization:
            raw_quality, raw_time = raw_state
            quality_bounds = np.linspace(0, 1, self.QUALITY_CLASS_COUNT)
            time_bounds = np.linspace(
                0, 100, self.TIME_CLASS_COUNT)
            return utils.digitize(raw_quality, quality_bounds), utils.digitize(raw_time, time_bounds)

        return raw_state

    def get_solution_quality(self):
        cost = self.mem['cost']
        quality = self.start_heuristic / cost
        return quality

    def get_time(self):
        self.mem['time'] = tm.time() - self.mem['start_time']
        return self.mem['time']

    def seed(self, seed):
        super().seed(seed)
        self.problem.seed(seed)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
