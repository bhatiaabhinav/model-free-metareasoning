# import itertools
# import os
# import re
# import sys
import time as tm
from multiprocessing import Manager

import gym
import numpy as np

from MFMR import utils
from MFMR.algos.search_problem import (Node, PriorityDict, SearchProblem,
                                       get_key)
from MFMR.async_algo import AsyncAlgo
from MFMR.utils import update_mean_std_corr

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
        self.w = weight
        self.discretization = discretization
        if self.discretization:
            raise ValueError(
                'Discretization is not yet supported for this environment')

    # ------------------------------------------ helper methods -----------------------------------------------

    # ----------------------------------------------------------------------------------------------------------

    def reset(self):
        self.problem.reset()
        self.mem['cost'] = np.inf
        self.start_heuristic = self.problem.heuristic(self.problem.start_state)
        self.mem['start_time'] = tm.time()
        self.mem['time'] = 0
        self.mem['interrupted'] = False
        '''
        The following stats represent (E[G], E[H], Std(G), Std(H), Corr(G, H), n).

        The reason for chosing to maintain these specific stats is that other important stats
        can be recovered in these terms:

        E[F] = E[G + H] = E[G] + E[H]
        E[Weighted F] = E[G + wH] = E[G] + w * E[H]
        Var(F) = Var(G) + Var(H) + 2 * Cov(G, H) = Var(G) + Var(H) + 2 * Corr(G, H) * Std(G) * Std(H)
        Var(Weighted F) = Var(G) + w^2 * Var(H) + 2w * Cov(G, H)

        The reason we cannot directly maintain E[G + wH] & Std(G + wH) is because
        as w changes, these stats will get invalidated.
        '''
        self.mem['stats_open'] = (0, 0, 0, 0, 0, 0)
        self.mem['stats_closed'] = (0, 0, 0, 0, 0, 0)

    def run(self):
        problem = self.problem

        start_node = Node(problem.start_state)
        start_node_key = get_key(start_node.state)

        '''The open list prioritizes nodes acc to f' = g + wh '''
        open_list = PriorityDict(tie_breaker='LIFO')
        closed_set = set()

        '''The upperbound is the f = g + h of the latest solution found'''
        best_solution_f = float('inf')
        '''The lowerbound is the lowest f = g + h among the nodes in the open list'''
        # lowest_open_f = 0

        '''add root node to open list'''
        start_node_g, start_node_h = 0.0, problem.heuristic(start_node.state)
        open_list.add(start_node_key, start_node_g +
                      self.w * start_node_h, start_node)
        # lowest_open_f = start_node_g + start_node_h
        # self.mem['stats_open'] = update_mean_std_corr(
        #     *self.mem['stats_open'], start_node_g, start_node_h)

        path_costs = {start_node_key: start_node.path_cost}

        while open_list:
            _, current_node = open_list.pop()
            current_node_key = get_key(current_node.state)
            current_node_g, current_node_h = current_node.path_cost, problem.heuristic(
                current_node.state)
            current_node_f = current_node_g + current_node_h
            # self.mem['stats_open'] = update_mean_std_corr(
            #     *self.mem['stats_open'], current_node_g, current_node_h, remove=True)

            if current_node_f < best_solution_f:
                '''This node is worth exploring. Its subtree might contain a better solution'''
                closed_set.add(current_node_key)
                # self.mem['stats_closed'] = update_mean_std_corr(
                #     *self.mem['stats_closed'], current_node_g, current_node_h)

                for child_node in problem.get_children_nodes(current_node):
                    child_node_g, child_node_h = child_node.path_cost, problem.heuristic(
                        child_node.state)
                    child_node_f = child_node_g + child_node_h
                    child_node_key = get_key(child_node.state)

                    if child_node_f < best_solution_f:
                        '''This child is worthy of creation. Its subtree might create a better solution'''
                        if problem.goal_test(child_node.state):
                            '''We found a solution. Now we won't add this node to any list.
                            We want to prune this subtree since we are not going to get any better solutions down this lane'''
                            path_costs[child_node_key] = child_node.path_cost
                            best_solution_f = child_node_f
                            self.mem['solution'] = child_node.get_solution()
                            self.mem['cost'] = child_node.path_cost
                        elif child_node_key in closed_set or child_node_key in open_list:
                            '''okay.. we have seen this state before. This child is so unoriginal: a duplicate.'''
                            if path_costs[child_node_key] > child_node.path_cost:
                                '''This duplicate child is better than the old node with this state. The operation below
                                removes old node with g=`path_costs[child_node_key]` from either closed set or open list.
                                Then adds a new node with same state as the removed node, but with g=`child_node_g`, to the open list.
                                The statistics need to be updated accordingly.'''
                                old_node_g = path_costs[child_node_key]
                                old_node_h = child_node_h

                                '''the replacement operation:'''
                                path_costs[child_node_key] = child_node.path_cost
                                if child_node_key in closed_set:
                                    closed_set.remove(child_node_key)
                                    '''stats update due to remove old node from closed set:'''
                                    # self.mem['stats_closed'] = update_mean_std_corr(
                                    #     *self.mem['stats_closed'], old_node_g, old_node_h, remove=True)
                                else:
                                    del open_list[child_node_key]
                                    '''stats update due to remove old node from open list:'''
                                    # self.mem['stats_open'] = update_mean_std_corr(
                                    #     *self.mem['stats_open'], old_node_g, old_node_h, remove=True)

                                open_list.add(
                                    child_node_key, child_node_g + self.w * child_node_h, child_node)

                                '''stats update due to add new node to open list:'''
                                # lowest_open_f = min(lowest_open_f, )
                                # self.mem['stats_open'] = update_mean_std_corr(
                                #     *self.mem['stats_open'], child_node_g, child_node_h)
                            else:
                                '''ignore this duplicate child node because it is worse than the old node'''
                                pass
                        else:
                            '''this is a new non-goal child node, and its subtree is worth exploring
                            (because its f value is less than any solution known so far). So let's add it to open list'''
                            path_costs[child_node_key] = child_node_g
                            open_list.add(
                                child_node_key, child_node_g + self.w * child_node_h, child_node)
                            # self.mem['stats_open'] = update_mean_std_corr(
                            #     *self.mem['stats_open'], child_node_g, child_node_h)
                    else:
                        '''Abort this child because it does not have potential to improve the solution.
                        Ignore it - let's prune its subtree'''
                        pass

                    '''Done another iteration of the for loop : a child has been potentially added.
                    Let's check for interruption'''
                    self.mem['time'] = tm.time() - self.mem['start_time']
                    if self.mem['interrupted']:
                        break
                '''
                For loop ends here. We have expanded the current node and added (some of) its children to open list.
                '''
            else:
                '''This is a useless node in the openlist - no potential to improve the solution.
                Ignore it - let's prune its subtree'''
                pass

            '''Done another iteration of the while loop : processed a node from the open list.
            Let's check for interruption'''
            self.mem['time'] = tm.time() - self.mem['start_time']
            if self.mem['interrupted']:
                break

        '''
        At this point, EITHER the open list is empty and we have found an optimal solution
        OR the algorithm was interrupted and we have a suboptimal solution.
        '''

        # print('Done')

    def update_hyperparams(self, hyperparams):
        pass

    def get_hyperparam_space(self) -> gym.Space:
        return gym.spaces.Box(low=np.array([]), high=np.array([]))

    def interrupt(self):
        self.mem['interrupted'] = True

    def get_obs_space(self) -> gym.Space:
        # TODO: Incorporate statistics of closed set & open list.
        return gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.QUALITY_CLASS_COUNT, self.TIME_CLASS_COUNT]), shape=(2, ), dtype=np.float)

    def get_obs(self):
        # TODO: Incorporate statistics of closed set & open list.
        self.mem['time'] = tm.time() - self.mem['start_time']
        cost, time = self.mem['cost'], self.mem['time']
        quality = self.start_heuristic / cost
        return quality, time

    # not yet supported
    # def get_discretized_state(self, raw_state):
    #     if self.discretiz7ation:
    #         raw_quality, raw_time = raw_state
    #         quality_bounds = np.linspace(0, 1, self.QUALITY_CLASS_COUNT)
    #         time_bounds = np.linspace(
    #             0, 100, self.TIME_CLASS_COUNT)
    #         return utils.digitize(raw_quality, quality_bounds), utils.digitize(raw_time, time_bounds)

    #     return raw_state

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
