import itertools
import os
import re
import sys
import time as tm

import gym
import numpy as np

from MFMR import utils
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


class Tsp(AsyncAlgo):
    QUALITY_CLASS_COUNT = 100
    TIME_CLASS_COUNT = 100

    def __init__(self, instances_directory, index_file_path, discretization):
        super().__init__()
        self.iterations = ITERATIONS
        self.discretization = discretization
        # List of Tuples: (cities, start_city, optimal_cost)
        self.problems = []
        optimal_costs = utils.get_csv_dictionary(index_file_path)
        for file in os.listdir(instances_directory):
            if file.endswith(".tsp"):
                instance = os.path.splitext(file)[0]
                file_path = os.path.join(instances_directory, file)
                cities, start_city = self.load_instance(file_path)
                optimal_cost = optimal_costs[instance]
                self.problems.append((cities, start_city, optimal_cost))

    # ------------------------------------------ helper methods -----------------------------------------------
    def get_initial_random_tour(self, states, start_state):
        adjustable_states = set(states) - set([start_state])
        return [start_state] + list(self.random.permutation(list(adjustable_states))) + [start_state]

    def get_swappable_cities(self, tour):
        return range(1, len(tour) - 1)

    def get_mutated_tour(self, tour, first_key, second_key):
        mutated_tour = list(tour)
        mutated_tour[first_key], mutated_tour[second_key] = mutated_tour[second_key], mutated_tour[first_key]
        return mutated_tour

    def get_distance(self, first_city, second_city):
        return np.linalg.norm(np.subtract(first_city, second_city))

    def get_tour_distance(self, tour):
        distance = 0
        limit = len(tour)
        for i in range(limit):
            if i + 1 == limit:
                break
            distance += self.get_distance(tour[i], tour[i + 1])
        return distance

    def get_graph(self, cities):
        graph = {}
        for start_city in cities:
            graph[start_city] = {}
            for end_city in cities:
                graph[start_city][end_city] = self.get_distance(
                    start_city, end_city)
        return graph

    def get_nearest_city_distance(self, start_city, cities):
        nearest_distance = float("inf")
        for city in cities:
            if start_city == city:
                continue
            current_city_distance = self.get_distance(start_city, city)
            if current_city_distance < nearest_distance:
                nearest_distance = current_city_distance
        return nearest_distance

    def get_mst_distance(self, start_city, cities):
        subset = cities - set([start_city])
        graph = self.get_graph(subset)

        predecessors = {}
        key = {}
        queue = {}

        for vertex in graph:
            predecessors[vertex] = -1
            key[vertex] = sys.maxsize

        key[start_city] = 0

        for vertex in graph:
            queue[vertex] = key[vertex]

        while queue:
            city = utils.pop(queue)
            for vertex in graph[city]:
                if vertex in queue and graph[city][vertex] < key[vertex]:
                    predecessors[vertex] = city
                    key[vertex] = graph[city][vertex]
                    queue[vertex] = graph[city][vertex]

        cost = 0
        for parent_city in predecessors:
            child_city = predecessors[parent_city]
            if child_city != -1:
                cost += self.get_distance(parent_city, child_city)
        return cost + 2 * self.get_nearest_city_distance(start_city, cities)

    def get_instance(self, size, start_position, end_position, minimum_distance):
        choices = np.arange(start_position, end_position, minimum_distance)
        cities = set()
        while len(cities) < size:
            x = round(self.random.choice(choices), 3)
            y = round(self.random.choice(choices), 3)
            cities.add((x, y))
        return cities

    def save_instance(self, name, cities):
        size = len(cities)
        node_coord_section = ""
        for i, city in enumerate(cities):
            x, y = city
            delimiter = DELIMITER if i < size - 1 else ""
            node_coord_section += CITY_TEMPLATE % (i + 1, x, y, delimiter)
        instance = FILE_TEMPLATE % (name, COMMENT, size, node_coord_section)
        f = open(name, "w")
        f.write(instance)
        f.close()

    def load_instance(self, filename):
        cities = set()
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                match = re.search(CITY_PATTERN, line)
                if match:
                    x = float(match.groups()[0])
                    y = float(match.groups()[1])
                    cities.add((x, y))
        start_city = list(cities)[0]
        return cities, start_city
    # ----------------------------------------------------------------------------------------------------------

    def reset(self):
        self.mem.clear()
        self.instance_id = self.random.randint(len(self.problems))
        self.cities, self.start_city, self.optimal_cost = self.problems[self.instance_id]
        self.mem['tour'] = self.get_initial_random_tour(
            self.cities, self.start_city)
        self.mem['cost'] = self.get_tour_distance(self.mem['tour'])
        self.mem['start_time'] = tm.time()
        self.mem['time'] = 0
        self.mem['interrupted'] = False

    def run(self):
        # print("Started")

        tour = self.mem['tour']
        distance = self.mem['cost']
        cities = self.get_swappable_cities(self.mem['tour'])
        for _ in range(self.iterations):
            has_changed = False
            best_tour = tour
            best_distance = distance
            for first_key, second_key in itertools.combinations(cities, 2):
                current_tour = self.get_mutated_tour(
                    tour, first_key, second_key)
                current_distance = self.get_tour_distance(current_tour)

                if current_distance < best_distance:
                    best_tour = current_tour
                    best_distance = current_distance
                    self.mem["tour"] = current_tour
                    self.mem["cost"] = best_distance

                    has_changed = True

                self.mem['time'] = tm.time() - self.mem['start_time']

                if self.mem['interrupted']:
                    break

            tour = best_tour
            distance = best_distance

            if not has_changed or self.mem['interrupted']:
                break

        # print("Finished")

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
        quality = self.optimal_cost / cost
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
        quality = self.optimal_cost / cost
        return quality

    def get_time(self):
        self.mem['time'] = tm.time() - self.mem['start_time']
        return self.mem['time']

    def render(self, mode='human'):
        pass

    def close(self):
        super().close()
