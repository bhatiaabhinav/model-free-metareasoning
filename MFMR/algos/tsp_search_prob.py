import logging
import sys
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import connect, disconnect, viridis
from numpy.core.multiarray import can_cast

from MFMR.algos.search_problem import SearchProblem

logger = logging.getLogger(__name__)
ldebug = logger.isEnabledFor(logging.DEBUG)


class TSPProblem(SearchProblem):
    def __init__(self, N_range, sparsity_range):
        super().__init__()
        self.N_range = N_range
        self.N = self.N_range[0] + \
            self.random.randint(self.N_range[1] + 1 - self.N_range[0])
        self.instance = None
        self.distances = None
        self.start_state = None
        self.cache = {}
        self.epsilon = 0.001
        self.sparsity_range = sparsity_range
        self.sparsity = 0

    def seed(self, seed):
        super().seed(seed)

    def get_obs(self):
        return [self.N / self.N_range[1], self.sparsity]
        # return []

    def connect(self, distances, city1, city2, min_distance, max_distance):
        distance = min_distance + self.random.rand() * (max_distance - min_distance)
        distances[city1, city2] = distance
        distances[city2, city1] = distance
        ldebug and logger.debug(f'{city1}---{distance}---{city2}')
        # print(f'{city1}---{distance}---{city2}')
        return distances

    def isConnected(self, distances, city1, city2):
        return distances[city1, city2] < np.inf

    def create_tsp(self, N, sparsity=1, min_distance=1, max_distance=50):
        distances = np.ones((N, N)) * np.inf
        num_connections = 0
        for city in range(N):
            distances[city, city] = 0
        for city in range(N-1):
            self.connect(distances, city, city + 1, min_distance, max_distance)
            num_connections += 1
        self.connect(distances, N-1, 0, min_distance, max_distance)
        num_connections += 1
        for city1 in range(N - 1):
            for city2 in range(city1 + 1, N):
                if not city1 == city2 and not self.isConnected(distances, city1, city2) and self.random.rand() > sparsity:
                    # Then connect
                    self.connect(distances, city1, city2,
                                 min_distance, max_distance)
                    num_connections += 1
        max_connections = self.N * (self.N - 1) // 2
        sparsity = (max_connections - num_connections) / max_connections
        # print(num_connections, max_connections)
        return distances, sparsity

        # def check_conflict(self, temp_instance, node):
        #     # print (node[0],node[1])
        #     # print (temp_instance)
        #     distance = list(map(lambda y: np.sqrt(
        #         (node[0] - y[0])**2 + (node[1] - y[1])**2), temp_instance))
        #     # print(distance)
        #     if(min(distance) > self.epsilon):
        #         return True
        #     return False

        # def connect_cities(self, i, j, )

        # def calculate_grid_distances(self):
        #     grid_distances = np.ones((self.N, self.N))

        #     for i in range(self.N):
        #         for j in range(self.N):
        #             tuple1 = self.instance[i]
        #             tuple2 = self.instance[j]

        #             grid_distances[i, j] = np.sqrt(
        #                 (tuple1[0] - tuple2[0])**2 + (tuple1[1] - tuple2[1])**2)
        #             grid_distances[j, i] = grid_distances[i, j]
        #     return grid_distances

    def reset(self):
        '''Add N (=chosen at random from N_options) random cities of the form (x, y) to instance where each city is at least a distance of epsilon from each other'''
        self.cache.clear()
        self.N = self.N_range[0] + \
            self.random.randint(self.N_range[1] + 1 - self.N_range[0])
        # instance = []
        # while (len(instance) < self.N):
        #     x_cord = self.random.uniform(0, 1)
        #     y_cord = self.random.uniform(0, 1)
        #     if(len(instance) == 0):
        #         instance.append((x_cord, y_cord))
        #     elif self.check_conflict(instance, (x_cord, y_cord)):
        #         instance.append((x_cord, y_cord))

        self.start_state = [0]
        self.distances, self.sparsity = self.create_tsp(
            self.N, sparsity=self.sparsity_range[0] + self.random.rand() * self.sparsity_range[1])
        self.info = {
            'tsp_n': self.N,
            'tsp_sparsity': self.sparsity
        }
        logging.getLogger(__name__).info(
            f'TSP N = {self.N}, Sparsity = {self.sparsity}')

    def get_children_nodes(self, parent):
        return super().get_children_nodes(parent)

    def goal_test(self, state):
        if len(state) == (self.N + 1):
            assert np.sum(state[:self.N]) == (
                self.N) * (self.N - 1) / 2, "City index sum not equal"
            assert state[0] == state[-1], "Fist last not equal"
            return True

    def successors(self, state):
        for a in self.operators(state):
            ns = self.successor_fn(state, a)
            yield {'state': ns, 'action': a}

    def cost(self, state, action, next_state):
        return self.distances[state[-1], action]

    def heuristic(self, state):
        # explored_nodes = len(s.city_list)
        key = self.hash_state(state)
        if key in self.cache:
            # print('cache hit for', state)
            return self.cache[key]
        unexplored_distances = self.distances
        N = self.N
        s = state
        mst_cities = np.array(list(range(N)))
        mst_del_cities = []

        # print("for state", state, "mst cities before: \n", mst_cities)

        if len(s) > 1:
            mst_del_cities = np.array(list(set(s[0:-1])))
            ldebug and logger.debug(f'mst del cities {mst_del_cities}')
            unexplored_distances = np.delete(
                unexplored_distances, mst_del_cities, 0)
            unexplored_distances = np.delete(
                unexplored_distances, mst_del_cities, 1)
            mst_cities = np.array(
                list(set(list(range(N))) - set(mst_del_cities)))
            # logger.debug(f'mst cities {mst_cities}')
            # print(f'mst cities {mst_cities}')

        # print("mst del cities \n",mst_del_cities)
        # print("mst cities final", mst_cities)
        # print("tree state: \n",s.city_list)
        # print("mst distances: \n",unexplored_distances)

        # print(unexplored_distances)
        if len(mst_cities) > 0:
            p = Prims(len(mst_cities), unexplored_distances)
            # r = p.calculate_mst()
            r2 = p.my_mst()
            # assert (r == np.inf and r2 == np.inf) or abs(
            # r - r2) < 0.0001, f"{r}, {r2}, {unexplored_distances}"
        else:
            r2 = 0.0

        self.cache[key] = r2
        return r2

    def hash_state(self, state):
        return str(state)

    def operators(self, s):
        # print(s, self.N)
        if len(s) == self.N:
            options = [0] if self.distances[0, s[-1]] < np.inf else []
            # options = [0]
        else:
            visited = set(s)
            can_visit = set(np.where(self.distances[s[-1]] < np.inf)[0])
            # can_visit = set(range(self.N))
            options = list(can_visit - visited)
        ldebug and logger.debug(f'options from city {s[-1]}={options}')
        return options

    def successor_fn(self, s, a):
        return s + [a]


class Prims:
    # N : number of vertices
    # distance : 2D np array with euclidean distance between 2 nodes
    def __init__(self, N, distances):
        # logger.debug(f'Prims N = {N}')
        self.N = N
        self.grid_distances = distances
        self.mst_vertices = np.zeros((N))
        self.mst_parents = np.zeros((N))
        self.mst_node_values = np.array(
            [0] + [np.inf] * (N - 1), np.float32)

        self.mst_parents[0] = -1
        self.result = []
        self.cost = 0
        # print(self.grid_distances, "\n\n", self.mst_vertices,
        #       "\n\n", self.mst_parents, "\n\n", self.mst_node_values)

    def find_min_cutedge(self):
        min_cutedge = np.inf
        min_cutindex = -1

        for i in range(0, self.N):
            if (self.mst_vertices[i] == 0) & (min_cutedge > self.mst_node_values[i]):
                min_cutindex = i
                min_cutedge = self.mst_node_values[i]

        return min_cutindex, min_cutedge

    def adjacent_nodes(self, node):
        d = self.grid_distances[node]
        return set(np.where((0 < d) * (d < np.inf))[0])

    def any_disconnected_city(self):
        # logger.debug(self.grid_distances)
        return np.any(np.sum(np.isfinite(self.grid_distances), axis=1) == 1)

    def is_a_sequence(self):
        return np.all(np.sum(np.isfinite(self.grid_distances), axis=1) <= 3)

    def is_a_sequence(self):
        return np.all(np.sum(np.isfinite(self.grid_distances), axis=1) <= 3)

    def graph_is_disconnected(self):
        connected_to_0 = self.connected_to(0, self.grid_distances, set())
        # assert max(connected_to_0) < self.N
        # print(connected_to_0)
        return len(connected_to_0) < self.N

    def connected_to(self, node_from, distances, visited_set):
        # print(node_from, visited_set)
        visited_set.add(node_from)
        # print(node_from, adjacent_nodes)
        for node in self.adjacent_nodes(node_from):
            if not node in visited_set:
                self.connected_to(node, distances, visited_set)
        return visited_set

    def my_mst(self):
        def toSet(bitset):
            return set(np.where(bitset)[0])
        d = self.grid_distances
        mst_nodes = np.zeros(self.N)  # tells vertices included in mst
        key_vals = np.inf * np.ones(self.N)
        key_vals[0] = 0  # distance from 0 is 0

        # print("mst", toSet(mst_nodes), "key_vals", key_vals)
        while(sum(mst_nodes) < self.N):
            candidates = toSet(1 - mst_nodes)
            candidates_values = [key_vals[c] for c in candidates]
            best_node = min(zip(candidates, candidates_values),
                            key=lambda ct: ct[1])[0]
            # print('picking', best_node)
            mst_nodes[best_node] = 1
            adjacent_nodes = self.adjacent_nodes(best_node) - toSet(mst_nodes)
            # print('adjacent', adjacent_nodes)
            for adjacent in adjacent_nodes:
                key_vals[adjacent] = min(
                    key_vals[adjacent], d[best_node, adjacent])
            # print("mst", toSet(mst_nodes), "key_vals", key_vals)

        # print('done')
        cost = np.sum(key_vals)
        # print('cost', cost)
        return cost

    def calculate_mst(self):
        start = time.time()
        # print(self.grid_distances)
        # logger.debug('Calculating MST')
        # print('donig mst N = ', self.N)
        if self.N == 1:
            self.cost = 0
            # print('here')
        elif self.graph_is_disconnected():
            # print('yo')
            self.cost = np.inf
        elif self.is_a_sequence():
            # print('yep')
            distances = np.where(self.grid_distances ==
                                 np.inf, 0, self.grid_distances)
            # print(distances)
            self.cost = np.sum(distances) / 2
        else:
            while np.sum(self.mst_vertices) < self.N:
                node, min_cutedge = self.find_min_cutedge()
                # print("\n\nNode: ",node)
                if node == -1 or min_cutedge == np.inf:
                    print('here', self.N, np.sum(self.mst_vertices))
                    assert False, 'This should not happen'
                    break
                self.mst_vertices[node] = 1
                self.cost += self.mst_node_values[node]
                # logger.debug(f'node={node}, cost={self.cost}')

                # print("MST node values before: ",self.mst_node_values,"\n\n")

                for child in range(0, self.N):
                    if ((node != child) & (self.mst_vertices[child] == 0) & (self.grid_distances[node, child] < self.mst_node_values[child])):
                        self.mst_parents[child] = node
                        self.mst_node_values[child] = self.grid_distances[node, child]
                # print("MST node values after: ",self.mst_node_values,"\n\n")
                #print("MST Parents: ", self.mst_parents)
        total = time.time() - start
        if total > 5:
            print(total)
        # print(f'MST = {self.cost}')
        ldebug and logger.debug(f'MST = {self.cost}')
        return self.cost
