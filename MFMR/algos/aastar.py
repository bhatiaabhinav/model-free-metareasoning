import logging
import time as tm
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
ldebug = logger.isEnabledFor(logging.DEBUG)
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
        if f < np.inf:
            self.stats = update_mean_std_corr(*self.stats, g, h, remove=True)
        return key, node, g, h

    def peek(self, w_index):
        """Peeks the node with the least f' value from the open list with the given w_index. Return key, (node, g, h), f'"""
        key, (node, g, h), f = self.open_lists[w_index].peek()
        # print('least right now', node.state, g, h, g + h)
        return key, (node, g, h), f

    def delete(self, key):
        '''deletes the node with with given key from all the open lists.
        Does nothing if the key is not present
        '''
        if key in self:
            del self[key]

    def add(self, key, node, g, h):
        '''Adds a key-node pair to each open list with f' priority calculated acc to respective weights'''
        # print('adding', node.state, g, h, g + h)
        for i, w in enumerate(self.ws):
            f = g + w * h
            self.open_lists[i].add(key, (node, g, h), f)
        if g + h < np.inf:
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
        # print('deleting', node.state, g, h, g + h)
        for i, w in enumerate(self.ws):
            del self.open_lists[i][key]
        if f < np.inf:
            self.stats = update_mean_std_corr(*self.stats, g, h, remove=True)

    def __len__(self):
        return len(self.open_lists[self.w_index])


class AAstar(AsyncAlgo):
    QUALITY_CLASS_COUNT = 100
    TIME_CLASS_COUNT = 100

    def __init__(self, weight, weight_max, weight_interval, time_max, ref_nodes_budget, simulate_ref_machine, adjust_weight, observe_ub, search_problem_cls, *search_problem_args, **search_problem_kwargs):
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
        self.start_weight = weight
        self.t_max = time_max
        self.ref_nodes_budget = ref_nodes_budget
        self.simulate_ref_machine = simulate_ref_machine
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
    def log_num_nodes(self):
        return self.mem['log_num_nodes']

    @property
    def n_solutions(self):
        return self.mem['n_solutions']

    # def update_stats_(self, open_lists: MultiWeightOpenLists):
    #     # logger.debug('Recording stats')
    #     # print('recording stats')
    #     if len(open_lists) > 0:
    #         best_key, (best_node, best_g, best_h), best_f = open_lists.peek(0)
    #         # assert best_h == self.problem.heuristic(best_node.state)
    #         # assert best_g == best_node.path_cost
    #         assert abs(best_f - (best_g + best_h)) < 1e-6
    #         # print('new lower bound', best_f)
    #         assert best_f >= self.mem['cost_lb'] - \
    #             1e-6, f"{best_f}, {self.mem['cost_lb']}"
    #         self.mem['cost_lb'] = best_f if self.mem['cost'] > self.mem['cost_lb'] else self.mem['cost']
    #         self.mem['best_g'] = best_g
    #         self.mem['best_h'] = best_h
    #         self.mem['mean_g'], self.mem['mean_h'], self.mem['std_g'], self.mem['std_h'], self.mem['corr_gh'], n = open_lists.stats
    #         self.mem['frac_nodes'] = open_lists.get_frac_nodes()
    #     else:
    #         self.mem['cost_lb'] = self.mem['cost']
    #         self.mem['best_g'] = 0
    #         self.mem['best_h'] = 0
    #         self.mem['mean_g'], self.mem['mean_h'], self.mem['std_g'], self.mem['std_h'], self.mem['corr_gh'], _ = 0, 0, 0, 0, 0, 0
    #         self.mem['frac_nodes'] = 0

    def reset_stats(self, start_heuristic):
        stats = {
            'mean_g': 0,
            'mean_h': 0,
            'std_g': 0,
            'std_h': 0,
            'best_g': 0,
            'best_h': 0,
            'corr_gh': 0,
            'cost_lb': start_heuristic,
            'cost': np.inf,
            'n_solutions': 0,
            'log_open_nodes': 0,
            'nodes_expanded': 0,
            'time': 0,
            'wall_time': 0,
            'simulated_time': 0,
            'nodes_to_first_solution': self.ref_nodes_budget,
        }
        return stats

    def update_stats(self, stats, open_lists, wall_time, nodes_expanded):
        stats['nodes_expanded'] = nodes_expanded
        wall_expansion_rate = nodes_expanded / wall_time
        stats['wall_time'] = wall_time
        ref_expansion_rate = self.ref_nodes_budget / self.t_max
        stats['simulated_time'] = (wall_expansion_rate /
                                   ref_expansion_rate) * wall_time
        stats['time'] = stats['simulated_time'] if self.simulate_ref_machine else stats['wall_time']

        if len(open_lists) > 0:
            # open lists stats:
            best_key, (best_node, best_g, best_h), best_f = open_lists.peek(0)
            assert abs(best_f - (best_g + best_h)) < 1e-6
            assert best_f >= stats['cost_lb'] - \
                1e-6, f"{best_f},f{stats['cost_lb']}"
            stats['cost_lb'] = best_f if stats['cost'] > stats['cost_lb'] else stats['cost']
            stats['best_g'] = best_g
            stats['best_h'] = best_h
            stats['mean_g'], stats['mean_h'], stats['std_g'], stats['std_h'], stats['corr_gh'], _ = open_lists.stats
            stats['log_open_nodes'] = np.log10(len(open_lists))
        else:
            stats['cost_lb'] = stats['cost']
        return stats

    def write_stats(self, stats: dict):
        # for k, v in stats.items():
        #     self.mem[k] = v
        self.mem.update(stats)

    def reset(self):
        logger.info('Resetting')
        self.mem.clear()
        self.problem.reset()
        self.mem['w_index'] = self.w_init_index
        self.mem['action'] = 0
        self.start_time = tm.time()
        self.mem['interrupted'] = False
        self.start_heuristic = self.problem.heuristic(self.problem.start_state)
        stats = self.reset_stats(self.start_heuristic)
        self.write_stats(stats)

    def stop_condition(self, stats, wall_time, nodes_expanded):
        # print(self.cost_lb, self.cost_ub)
        converged = stats['cost_lb'] >= stats['cost']
        if converged:
            stats['cost_lb'] = stats['cost']
        wall_expansion_rate = nodes_expanded / wall_time
        ref_expansion_rate = self.ref_nodes_budget / self.t_max
        simulated_time = (wall_expansion_rate / ref_expansion_rate) * wall_time
        time = simulated_time if self.simulate_ref_machine else wall_time
        ldebug and logger.debug(
            f'Checking for stop. interrupt={self.mem["interrupted"]}, converged={converged}, time={time}')
        ans = converged or self.mem['interrupted'] or (time > self.t_max)
        ldebug and logger.debug(f'Stop={ans}')
        return ans

    def run(self):
        logger.info('Starting run')
        # stats
        start_wall_time = tm.time()
        nodes_expanded = 0
        stats = self.reset_stats(self.start_heuristic)

        problem = self.problem
        start_node = Node(problem.start_state)
        start_node_key = problem.hash_state(start_node.state)

        ldebug and logger.debug('Creating MultiWeight open lists')
        '''The open list prioritizes nodes acc to f' = g + wh '''
        open_lists = MultiWeightOpenLists(
            self.ws[self.w_init_index], w_min=self.w_min, w_max=self.w_max, w_interval=self.w_interval)

        closed_set = set()

        '''The upperbound is the f = g + h of the latest solution found'''
        best_solution_f = np.inf

        '''add root node to open list'''
        start_node_g, start_node_h = 0.0, problem.heuristic(start_node.state)
        open_lists.add(start_node_key, start_node, start_node_g, start_node_h)
        ldebug and logger.debug(
            f'Added start node{start_node.state}. f={start_node_h}')
        stats = self.update_stats(
            stats, open_lists, tm.time() - start_wall_time, nodes_expanded)
        self.write_stats(stats)

        path_costs = {start_node_key: start_node.path_cost}

        while len(open_lists) > 0:
            '''set open list weight to what the metareasoner wants'''
            open_lists.w_index = self.w_index
            current_node_key, current_node, current_node_g, current_node_h = open_lists.pop()
            current_node_f = current_node_g + current_node_h
            ldebug and logger.debug(
                f'popped {current_node.state} f={current_node_f}')

            if current_node_f < best_solution_f:
                ldebug and logger.debug(f'node worth exploring')
                '''This node is worth exploring. Its subtree might contain a better solution'''
                closed_set.add(current_node_key)

                for child_node in problem.get_children_nodes(current_node):
                    child_node_g, child_node_h = child_node.path_cost, problem.heuristic(
                        child_node.state)
                    child_node_f = child_node_g + child_node_h
                    ldebug and logger.debug(
                        f'child {child_node.state} f={child_node_f}')
                    child_node_key = problem.hash_state(child_node.state)

                    if child_node_f < best_solution_f:
                        ldebug and logger.debug('worthy child')
                        '''This child is worthy of creation. Its subtree might create a better solution'''
                        if problem.goal_test(child_node.state):
                            '''We found a solution. Now we won't add this node to any list.
                            We want to prune this subtree since we are not going to get any better solutions down this lane'''
                            ldebug and logger.debug('goal child')
                            path_costs[child_node_key] = child_node.path_cost
                            if best_solution_f == np.inf:
                                stats['nodes_to_first_solution'] = nodes_expanded
                            best_solution_f = child_node_f
                            # self.mem['solution'] = child_node.get_solution()
                            stats['cost'] = child_node_f
                            stats['n_solutions'] = stats['n_solutions'] + 1
                            self.write_stats(stats)
                            # print('solution, cost = ', child_node_f)
                        elif child_node_key in closed_set or child_node_key in open_lists:
                            '''okay.. we have seen this state before. This child is so unoriginal: a duplicate.'''
                            if path_costs[child_node_key] > child_node.path_cost:
                                ldebug and logger.debug('Good duplicate')
                                '''This duplicate child is better than the old node with this state. The operation below
                                removes old node with g=`path_costs[child_node_key]` from either closed set or open list.
                                Then adds a new node with same state as the removed node, but with g=`child_node_g`, to the open list.
                                The statistics need to be updated accordingly.'''
                                # old_node_g = path_costs[child_node_key]
                                # old_node_h = child_node_h

                                '''the replacement operation:'''
                                path_costs[child_node_key] = child_node.path_cost
                                if child_node_key in closed_set:
                                    ldebug and logger.debug(
                                        'Removed old from closed set')
                                    closed_set.remove(child_node_key)
                                else:
                                    ldebug and logger.debug(
                                        'Removed old from open list')
                                    del open_lists[child_node_key]

                                ldebug and logger.debug(
                                    'Added duplicate to open list')
                                open_lists.add(
                                    child_node_key, child_node, child_node_g, child_node_h)
                            else:
                                ldebug and logger.debug('bad child')
                                '''ignore this duplicate child node because it is worse than the old node'''
                                pass
                        else:
                            '''this is a new non-goal child node and it is not a duplicate, and its subtree is worth exploring
                            (because its f value is less than any solution known so far). So let's add it to open list'''
                            ldebug and logger.debug('Added child to open list')
                            path_costs[child_node_key] = child_node_g
                            open_lists.add(
                                child_node_key, child_node, child_node_g, child_node_h)
                    else:
                        '''Abort this child because it does not have potential to improve the solution.
                        Ignore it - let's prune its subtree'''
                        logger.debug('ignore child')
                        pass

                    '''Done another iteration of the for loop : a child has been potentially added.
                    Let's check for interruption or timeout'''
                    if self.stop_condition(stats, tm.time() - start_wall_time, nodes_expanded):
                        break
                '''
                For loop ends here. We have expanded the current node and added (some of) its children to open list.
                '''
                nodes_expanded += 1
            else:
                '''This is a useless node in the openlist - no potential to improve the solution.
                Ignore it - let's prune its subtree'''
                logger.debug('Non worthy open')
                pass

            # nodes_expanded += 1
            wall_time = tm.time() - start_wall_time

            stats = self.update_stats(
                stats, open_lists, wall_time, nodes_expanded)
            self.write_stats(stats)

            '''Done another iteration of the while loop : processed a node from the open list.
            Let's check for interruption or timeout'''
            if self.stop_condition(stats, wall_time, nodes_expanded):
                print(nodes_expanded, nodes_expanded / wall_time)
                break
        '''
        At this point, EITHER the open list is empty and we have found an optimal solution
        OR the algorithm was interrupted and we have a suboptimal solution.
        '''

        logger.info('Run done')
        # print('finish')

    def interrupt(self):
        self.mem['interrupted'] = True

    def get_obs_space(self) -> gym.Space:
        '''weight, q, ub, t, sys_usage_cpu, log_open_nodes, best_g, best_h, mean_g, mean_h, std_g, std_h, corr_gh, search prob obs'''
        return gym.spaces.Box(low=np.array([0.0] * 13 + [0.0] * len(self.problem.get_obs())), high=np.array([1.0] * 13 + [1.0] * len(self.problem.get_obs())), dtype=np.float32)

    def get_obs(self):
        ldebug and logger.debug('getting obs')
        w = (self.w - self.w_min) / (self.w_max - self.w_min)
        q = (self.start_heuristic + 1) / (self.mem['cost'] + 1)
        ub = (self.start_heuristic + 1) / (self.mem['cost_lb'] + 1)
        t = self.mem['time'] / self.t_max
        cpu = psutil.cpu_percent() / 100
        n = self.mem['log_open_nodes']
        best_g = (self.start_heuristic + 1) / (self.mem['best_g'] + 1)
        best_h = (self.start_heuristic + 1) / (self.mem['best_h'] + 1)
        mean_g = (self.start_heuristic + 1) / (self.mem['mean_g'] + 1)
        mean_h = (self.start_heuristic + 1) / (self.mem['mean_h'] + 1)
        std_g = (self.start_heuristic + 1) / (self.mem['std_g'] + 1)
        std_h = (self.start_heuristic + 1) / (self.mem['std_g'] + 1)
        corr_gh = self.mem['corr_gh']
        prob_obs = self.problem.get_obs()

        obs = np.asarray([w, q, ub, t, cpu, n, best_g, best_h, mean_g, mean_h,
                          std_g, std_h, corr_gh] + list(prob_obs), dtype=np.float32)
        ldebug and logger.debug('got obs')
        return obs

    def get_info(self):
        # w = self.w
        q = self.get_solution_quality()
        assert q <= 1 + 1e-6
        cost_lb = self.mem['cost_lb']
        cost = self.mem['cost']
        q_ub = self.start_heuristic / (cost_lb)
        assert q_ub <= 1 + \
            1e-6, f"{q}, {q_ub}, {self.start_heuristic}, {cost_lb}, {cost}"
        # assert q <= q_ub, f"{q}, {q_ub}"
        cpu = psutil.cpu_percent() / 100
        best_g = (self.start_heuristic + 1) / (self.mem['best_g'] + 1)
        best_h = (self.start_heuristic + 1) / (self.mem['best_h'] + 1)
        mean_g = (self.start_heuristic + 1) / (self.mem['mean_g'] + 1)
        mean_h = (self.start_heuristic + 1) / (self.mem['mean_h'] + 1)
        std_g = (self.start_heuristic + 1) / (self.mem['std_g'] + 1)
        std_h = (self.start_heuristic + 1) / (self.mem['std_g'] + 1)
        corr_gh = self.mem['corr_gh']
        # nodes_to_first_solution = self.mem['nodes_to_first_solution']
        info = {
            'w': self.w,
            'start_w': self.start_weight,
            'w_max': self.w_max,
            'w_min': self.w_min,
            'w_interval': self.w_interval,
            't_max': self.t_max,
            'ref_nodes_budget': self.ref_nodes_budget,
            'simulate_ref_machine': int(self.simulate_ref_machine),
            'adjust_weight': self.adjust_weight,
            'solution_quality': q,
            'time': self.mem['time'],
            'nodes_expanded': self.mem['nodes_expanded'],
            'nodes_to_first_solution': self.mem['nodes_to_first_solution'],
            'wall_time': self.mem['wall_time'],
            'simulated_time': self.mem['simulated_time'],
            'q_ub': q_ub,
            'cpu': cpu,
            'n_solutions': self.mem['n_solutions'],
            'best_g': best_g,
            'best_h': best_h,
            'mean_g': mean_g,
            'mean_h': mean_h,
            'std_g': std_g,
            'std_h': std_h,
            'corr_gh': corr_gh,
            'log_open_nodes': self.mem['log_open_nodes'],
        }
        # if nodes_to_first_solution < np.inf:
        #     info['nodes_to_first_solution'] = nodes_to_first_solution
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
        # print(self.cost)
        return self.start_heuristic / (self.cost)

    @property
    def is_time_virtual(self):
        return self.nodes_budget is not None

    def get_time(self):
        return self.mem['time']

    def get_wall_time(self):
        return self.mem['wall_time']

    def seed(self, seed):
        super().seed(seed)
        self.problem.seed(seed)

    def render(self, mode='human'):
        pass

    def close(self):
        super().close()


if __name__ == '__main__':
    from MFMR.algos.tsp_search_prob import TSPProblem
    aastar = AAstar(2.0, 3.0, 0.1, 180, True, True, TSPProblem,

                    N_range=[10, 10], sparsity_range=[0, 0.0])
    aastar.seed(0)
    aastar.reset()
    aastar.run()
