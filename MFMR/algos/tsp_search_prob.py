import logging
import sys
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from MFMR.algos.search_problem import SearchProblem


class TSPProblem(SearchProblem):
    def __init__(self, N_options):
        super().__init__()
        self.N_options = N_options
        self.N = self.random.choice(self.N_options)
        self.instance = None
        self.grid_distances = None
        self.start_state = None
        self.cache = {}
        self.epsilon = 0.001

    def seed(self, seed):
        super().seed(seed)

    def check_conflict(self, temp_instance, node):
        # print (node[0],node[1])
        # print (temp_instance)
        distance = list(map(lambda y: np.sqrt(
            (node[0] - y[0])**2 + (node[1] - y[1])**2), temp_instance))
        # print(distance)
        if(min(distance) > self.epsilon):
            return True
        return False

    def calculate_grid_distances(self):
        grid_distances = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                tuple1 = self.instance[i]
                tuple2 = self.instance[j]

                grid_distances[i, j] = np.sqrt(
                    (tuple1[0] - tuple2[0])**2 + (tuple1[1] - tuple2[1])**2)
                grid_distances[j, i] = grid_distances[i, j]
        return grid_distances

    def reset(self):
        '''Add N (=chosen at random from N_options) random cities of the form (x, y) to instance where each city is at least a distance of epsilon from each other'''
        self.N = self.random.choice(self.N_options)
        logging.getLogger(__name__).info(f'TSP N = {self.N}')
        instance = []
        while (len(instance) < self.N):
            x_cord = self.random.uniform(0, 1)
            y_cord = self.random.uniform(0, 1)
            if(len(instance) == 0):
                instance.append((x_cord, y_cord))
            elif self.check_conflict(instance, (x_cord, y_cord)):
                instance.append((x_cord, y_cord))

        self.start_state = [0]
        self.instance = instance
        self.grid_distances = self.calculate_grid_distances()
        self.info = {
            'tsp_n': self.N
        }

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
        return self.grid_distances[state[-1], action]

    def heuristic(self, state):
        # explored_nodes = len(s.city_list)
        key = self.hash_state(state)
        if key in self.cache:
            return self.cache[key]
        unexplored_grid_distances = self.grid_distances
        N = self.N
        s = state
        mst_cities = np.array(list(range(N)))
        mst_del_cities = []

        # print("mst cities: \n",mst_cities)

        if len(s) > 2:
            mst_del_cities = np.array(list(set(s[1:-1])))
            unexplored_grid_distances = np.delete(
                unexplored_grid_distances, mst_del_cities, 0)
            unexplored_grid_distances = np.delete(
                unexplored_grid_distances, mst_del_cities, 1)
            mst_cities = np.array(
                list(set(list(range(N))) - set(mst_del_cities)))

        # print("mst del cities \n",mst_del_cities)
        # print("mst cities: \n",mst_cities)
        # print("tree state: \n",s.city_list)
        # print("mst distances: \n",unexplored_grid_distances)

        mst_nodes = len(mst_cities)
        p = Prims(mst_nodes, unexplored_grid_distances)
        r = p.calculate_mst()
        self.cache[key] = r

        return r

    def hash_state(self, state):
        return str(state)

    def operators(self, s):
        if len(s) == self.N:
            return [0]
        else:
            return list(set(list(range(self.N))) - set(s))

    def successor_fn(self, s, a):
        return s + [a]


class Prims:
    # N : number of vertices
    # distance : 2D np array with euclidean distance between 2 nodes
    def __init__(self, N, distances):
        self.N = N
        self.grid_distances = distances
        self.mst_vertices = np.zeros((N))
        self.mst_parents = np.zeros((N))
        self.mst_node_values = np.array(
            [0] + [sys.maxsize] * (N - 1), np.float32)

        self.mst_parents[0] = -1
        self.result = []
        self.cost = 0
        # print(self.grid_distances, "\n\n", self.mst_vertices,
        #       "\n\n", self.mst_parents, "\n\n", self.mst_node_values)

    def find_min_cutedge(self):
        min_cutedge = sys.maxsize
        min_cutindex = -1

        for i in range(0, self.N):
            if (self.mst_vertices[i] == 0) & (min_cutedge > self.mst_node_values[i]):
                min_cutindex = i
                min_cutedge = self.mst_node_values[i]

        return min_cutindex

    def calculate_mst(self):
        while np.sum(self.mst_vertices) < self.N:
            node = self.find_min_cutedge()
            # print("\n\nNode: ",node)
            self.mst_vertices[node] = 1
            self.cost += self.mst_node_values[node]

            # print("MST node values before: ",self.mst_node_values,"\n\n")

            for child in range(0, self.N):
                if ((node != child) & (self.mst_vertices[child] == 0) & (self.grid_distances[node, child] < self.mst_node_values[child])):
                    self.mst_parents[child] = node
                    self.mst_node_values[child] = self.grid_distances[node, child]

            # print("MST node values after: ",self.mst_node_values,"\n\n")

        #print("MST Parents: ", self.mst_parents)

        return self.cost
