
import logging
import time as tm
from logging import setLogRecordFactory
from typing import List

import gym
import numpy as np
import psutil

from MFMR.algos.search_problem import Node, PriorityDict, SearchProblem
from MFMR.async_algo import AsyncAlgo
from MFMR.utils import update_mean_std_corr

# from multiprocessing import Manager


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


logger = logging.getLogger(__name__)
ROUND = 2


class MultiWeightOpenLists:
    def __init__(self, w_init, w_min, w_max, w_interval=0.1):
        w = np.round(w_init, ROUND)
        self.ws = MultiWeightOpenLists.get_weights_list(
            w_min, w_max, w_interval)[-1]
        w_indexes = np.where(self.ws == w)[0]
        if len(w_indexes) == 0:
            raise ValueError(
                f'w_init={w} is not valid. It should be in the space created by other arguments')
        self.w_index = w_indexes[0]
        self.open_lists = [PriorityDict(tie_breaker='LIFO') for w in self.ws]
        # [g mean, h mean, g std, h std, corr, n]
        self.stats = [0, 0, 0, 0, 0, 0]

    @property
    def w(self):
        return self.ws[self.w_index]

    def get_w_index(w, ws):
        w = np.round(w, ROUND)
        ws = np.round(ws, ROUND)
        return np.where(ws == w)[0][0]

    def get_weights_list(w_min, w_max, w_interval):
        w_min = np.round(w_min, ROUND)
        w_max = np.round(w_max, ROUND)
        w_interval = np.round(w_interval, ROUND)
        ws = np.round(np.arange(w_min, w_max + w_interval, w_interval), ROUND)
        return w_min, w_max, w_interval, ws

    def pop(self):
        '''Retrives and delete the node with the least f' value from the open list with current w.
        Deletes the node from all the other open lists'''
        key, (node, g, h), f = self.open_lists[self.w_index].pop()
        for i, w in enumerate(self.ws):
            if not i == self.w_index:
                del self.open_lists[i][key]
        self.stats = update_mean_std_corr(*self.stats, g, h, remove=True)
        return key, node

    def peek(self, w_index):
        """Peeks the node with the least f' value from the open list with the given w_index. Return key, (node, g, h), f'"""
        return self.open_lists[w_index].peek()

    def delete(self, key):
        '''deletes the node with with given key from all the open lists.
        Does nothing if the key is not present
        '''
        if key in self:
            del self[key]

    def add(self, key, node, g, h):
        '''Adds a key-node pair to each open list with f' priority calculated acc to respective weights'''
        for i, w in enumerate(self.ws):
            f = g + w * h
            self.open_lists[i].add(key, (node, g, h), f)
        self.stats = update_mean_std_corr(*self.stats, g, h)

    def increment_w(self):
        '''Attempts to increment the current weight by w_interval, as long as it does not violate w_max'''
        if self.w_index < len(self.ws) - 1:
            self.w_index += 1

    def decrement_w(self):
        '''Attempts to decrement the current weight by w_interval, as long as it does not violate w_min'''
        if self.w_index > 0:
            self.w_index -= 1

    def get_lower_bound_stats(self):
        return [self.peek(i)[-1] for i, w in enumerate(self.ws)]

    def get_mean_stats(self):
        return [self.open_lists[i].mean for i, w in enumerate(self.ws)]

    def get_std_stats(self):
        return [self.open_lists[i].std for i, w in enumerate(self.ws)]

    def get_frac_nodes(self):
        return self.open_lists[self.w_index].frac_nodes

    def __contains__(self, key):
        return key in self.open_lists[self.w_index]

    def __delitem__(self, key):
        (node, g, h), f = self.open_lists[0][key]
        for i, w in enumerate(self.ws):
            del self.open_lists[i][key]
        self.stats = update_mean_std_corr(*self.stats, g, h, remove=True)

    def __len__(self):
        return len(self.open_lists[self.w_index])


class AAstar(AsyncAlgo):
    QUALITY_CLASS_COUNT = 100
    TIME_CLASS_COUNT = 100

    def __init__(self, weight, weight_max, weight_interval, time_max, adjust_weight, observe_ub, search_problem_cls, *search_problem_args, **search_problem_kwargs):
        super().__init__()
        self.iterations = ITERATIONS
        self.problem = search_problem_cls(
            *search_problem_args, **search_problem_kwargs)  # type: SearchProblem
        self.w_min = 1
        self.w_max = weight_max
        self.w_interval = weight_interval
        self.ws = MultiWeightOpenLists.get_weights_list(
            self.w_min, self.w_max, self.w_interval)[-1]
        self.w_init_index = MultiWeightOpenLists.get_w_index(weight, self.ws)
        self.t_max = time_max
        self.adjust_weight = adjust_weight
        self.observe_ub = observe_ub
        self.viewer = None

    @property
    def w_index(self):
        return self.mem['w_index']

    @property
    def w(self):
        return self.ws[self.w_index]

    @property
    def cost(self):
        return self.mem['cost']

    @property
    def cost_lb(self):
        return self.mem['cost_lb']

    @property
    def cost_ub(self):
        return self.cost

    @property
    def interrupted(self):
        return self.mem['interrupted']

    @property
    def frac_open_nodes(self):
        return self.mem['frac_nodes']

    @property
    def n_solutions(self):
        return self.mem['n_solutions']

    def update_stats(self, open_lists: MultiWeightOpenLists):
        best_key, (best_node, best_g, best_h), best_f = open_lists.peek(0)
        assert best_f == best_g + best_h
        self.mem['cost_lb'] = best_f
        self.mem['best_g'] = best_g
        self.mem['best_h'] = best_h
        self.mem['mean_g'], self.mem['mean_h'], self.mem['std_g'], self.mem['std_h'], self.mem['corr_gh'], n = open_lists.stats
        self.mem['frac_nodes'] = open_lists.get_frac_nodes()

    def reset(self):
        logger.info('Resetting')
        self.mem.clear()
        self.problem.reset()
        self.n = self.problem
        self.mem['w_index'] = self.w_init_index
        self.mem['action'] = 0
        self.mem['cost'] = np.inf
        self.start_heuristic = self.problem.heuristic(self.problem.start_state)
        self.start_time = tm.time()
        self.mem['interrupted'] = False
        self.mem['cost_lb'] = self.start_heuristic
        self.mem['best_g'] = 0
        self.mem['best_h'] = self.start_heuristic
        self.mem['mean_g'], self.mem['mean_h'], self.mem['std_g'], self.mem['std_h'], self.mem['corr_gh'] = 0, self.start_heuristic, 0, 0, 0
        self.mem['frac_nodes'] = 0
        self.mem['n_solutions'] = 0

    def stop_condition(self):
        converged = self.cost_lb >= self.cost_ub
        return converged or self.mem['interrupted'] or (tm.time() - self.start_time) > self.t_max

    def run(self):
        logger.info('Starting run')
        problem = self.problem

        start_node = Node(problem.start_state)
        start_node_key = problem.hash_state(start_node.state)

        '''The open list prioritizes nodes acc to f' = g + wh '''
        open_lists = MultiWeightOpenLists(
            self.ws[self.w_init_index], w_min=self.w_min, w_max=self.w_max, w_interval=self.w_interval)

        closed_set = set()

        '''The upperbound is the f = g + h of the latest solution found'''
        best_solution_f = float('inf')

        '''add root node to open list'''
        start_node_g, start_node_h = 0.0, problem.heuristic(start_node.state)
        open_lists.add(start_node_key, start_node, start_node_g, start_node_h)
        self.update_stats(open_lists)
        # assert self.mem['cost_lbs'][0] == start_node_h

        path_costs = {start_node_key: start_node.path_cost}

        while len(open_lists) > 0:
            '''set open list weight to what the metareasoner wants'''
            open_lists.w_index = self.w_index
            current_node_key, current_node = open_lists.pop()
            current_node_g, current_node_h = current_node.path_cost, problem.heuristic(
                current_node.state)
            current_node_f = current_node_g + current_node_h
            if len(open_lists) > 0:
                self.update_stats(open_lists)

            if current_node_f < best_solution_f:
                '''This node is worth exploring. Its subtree might contain a better solution'''
                closed_set.add(current_node_key)

                for child_node in problem.get_children_nodes(current_node):
                    child_node_g, child_node_h = child_node.path_cost, problem.heuristic(
                        child_node.state)
                    child_node_f = child_node_g + child_node_h
                    child_node_key = problem.hash_state(child_node.state)

                    if child_node_f < best_solution_f:
                        '''This child is worthy of creation. Its subtree might create a better solution'''
                        if problem.goal_test(child_node.state):
                            '''We found a solution. Now we won't add this node to any list.
                            We want to prune this subtree since we are not going to get any better solutions down this lane'''
                            path_costs[child_node_key] = child_node.path_cost
                            best_solution_f = child_node_f
                            # self.mem['solution'] = child_node.get_solution()
                            self.mem['cost'] = child_node_f
                            self.mem['n_solutions'] = self.mem['n_solutions'] + 1
                        elif child_node_key in closed_set or child_node_key in open_lists:
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
                                else:
                                    del open_lists[child_node_key]
                                    '''stats update due to remove old node from open list:'''

                                open_lists.add(
                                    child_node_key, child_node, child_node_g, child_node_h)
                                '''stats update due to add new node to open list:'''
                                self.update_stats(open_lists)
                            else:
                                '''ignore this duplicate child node because it is worse than the old node'''
                                pass
                        else:
                            '''this is a new non-goal child node and it is not a duplicate, and its subtree is worth exploring
                            (because its f value is less than any solution known so far). So let's add it to open list'''
                            path_costs[child_node_key] = child_node_g
                            open_lists.add(
                                child_node_key, child_node, child_node_g, child_node_h)
                            self.update_stats(open_lists)
                    else:
                        '''Abort this child because it does not have potential to improve the solution.
                        Ignore it - let's prune its subtree'''
                        pass

                    '''Done another iteration of the for loop : a child has been potentially added.
                    Let's check for interruption or timeout'''
                    if self.stop_condition():
                        break
                '''
                For loop ends here. We have expanded the current node and added (some of) its children to open list.
                '''
            else:
                '''This is a useless node in the openlist - no potential to improve the solution.
                Ignore it - let's prune its subtree'''
                pass

            '''Done another iteration of the while loop : processed a node from the open list.
            Let's check for interruption or timeout'''
            if self.stop_condition():
                break

        '''
        At this point, EITHER the open list is empty and we have found an optimal solution
        OR the algorithm was interrupted and we have a suboptimal solution.
        '''

        logger.info('Run done')

    def interrupt(self):
        self.mem['interrupted'] = True

    def get_obs_space(self) -> gym.Space:
        '''weight, c, t, sys_usage, frac_open_nodes, best_g, best_h, mean_g, mean_h, std_g, std_h, corr_gh, search prob obs'''
        return gym.spaces.Box(low=np.array([0.0] * 12 + [0.0]*len(self.problem.get_obs())), high=np.array([0.0] * 12 + [.0]*len(self.problem.get_obs())), dtype=np.float32)

    def get_obs(self):
        w = (self.w - self.w_min) / (self.w_max - self.w_min)
        q = (self.start_heuristic + 1) / (self.cost + 1)
        t = self.get_time() / self.t_max
        cpu = psutil.cpu_percent() / 100
        n = self.frac_open_nodes
        best_g = (self.start_heuristic + 1) / (self.mem['best_g'] + 1)
        best_h = (self.start_heuristic + 1) / (self.mem['best_h'] + 1)
        mean_g = (self.start_heuristic + 1) / (self.mem['mean_g'] + 1)
        mean_h = (self.start_heuristic + 1) / (self.mem['mean_h'] + 1)
        std_g = (self.start_heuristic + 1) / (self.mem['std_g'] + 1)
        std_h = (self.start_heuristic + 1) / (self.mem['std_g'] + 1)
        corr_gh = self.mem['corr_gh']
        prob_obs = self.problem.get_obs()

        return np.asarray([w, q, t, cpu, n, best_g, best_h, mean_g, mean_h, std_g, std_h, corr_gh] + list(prob_obs), dtype=np.float32)

    def get_info(self):
        w = self.w
        q = self.get_solution_quality()
        t = self.get_time()
        q_ub = self.start_heuristic / (self.cost_lb + 1e-6)
        cpu = psutil.cpu_percent() / 100
        n = self.frac_open_nodes
        best_g = (self.start_heuristic + 1) / (self.mem['best_g'] + 1)
        best_h = (self.start_heuristic + 1) / (self.mem['best_h'] + 1)
        mean_g = (self.start_heuristic + 1) / (self.mem['mean_g'] + 1)
        mean_h = (self.start_heuristic + 1) / (self.mem['mean_h'] + 1)
        std_g = (self.start_heuristic + 1) / (self.mem['std_g'] + 1)
        std_h = (self.start_heuristic + 1) / (self.mem['std_g'] + 1)
        corr_gh = self.mem['corr_gh']
        prob_obs = self.problem.get_obs()
        info = {'solution_quality': q, 'time': t, 'w': self.w, 'q_ub': q_ub, 'cpu': cpu,
                'n_solutions': self.n_solutions, 'best_g': best_g, 'best_h': best_h, 'mean_g': mean_g, 'std_g': std_g, 'mean_h': mean_h, 'std_h': std_h,  'corr_gh': corr_gh, 'frac_open_nodes': n}
        info.update(self.problem.info)
        return info

    def get_action_space(self) -> gym.Space:
        return gym.spaces.Discrete(len(self.get_action_meanings()))

    def get_action_meanings(self) -> List[str]:
        if self.adjust_weight:
            return ['NOOP', 'INC_W_COURSE', 'INC_W', 'DEC_W_COURSE', 'DEC_W']
        else:
            return ['NOOP', 'DUMMY_1', 'DUMMY_2', 'DUMMY_3', 'DUMMY_4']

    def set_action(self, action):
        super().set_action(action)
        meaning = self.get_action_meanings()[action]
        w_idx = self.w_index
        if meaning == 'INC_W_COURSE':
            w_idx += 4
        elif meaning == 'INC_W':
            w_idx += 1
        elif meaning == 'DEC_W_COURSE':
            w_idx -= 4
        elif meaning == 'DEC_W':
            w_idx -= 1
        elif meaning == 'NOOP':
            w_idx += 0
        elif meaning.startswith('DUMMY'):
            pass
        else:
            assert False, "Unknown Action? Did I check action meanings properly?"

        self.mem['w_index'] = min(max(0, w_idx), len(self.ws) - 1)

    def get_solution_quality(self):
        return self.start_heuristic / (self.cost + 1e-6)

    def get_time(self):
        time = tm.time() - self.start_time
        return time

    def seed(self, seed):
        super().seed(seed)
        self.problem.seed(seed)

    def render(self, mode='human'):
        pass

    def close(self):
        super().close()
