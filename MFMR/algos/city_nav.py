import logging
from typing import List, Set, Tuple

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


class Place:
    def __init__(self, parent_id_no, id_no: int, position: np.ndarray) -> None:
        self.position = position
        self.neighbors = set()  # type: Set[int]
        self.id_no = id_no
        self.parent_id_no = parent_id_no

    def __repr__(self) -> str:
        return f'{self.parent_id_no, self.id_no, self.position}'

    def __str__(self) -> str:
        return f'{self.parent_id_no, self.id_no}'


class City:
    def __init__(self, id_no: int, position: np.ndarray, n_p, random: np.random.RandomState, num_places=150, city_size=1) -> None:
        self.position = position
        self.id_no = id_no
        self.num_places = num_places
        self.n_p = n_p
        self.random = random
        self.city_size = city_size
        self._generate_places()
        self.neighbors = set()  # type: Set[int]

    def connect(self, place1: Place, place2: Place):
        if place2.id_no not in place1.neighbors:  # don't allow recalculation of cost
            distance = np.linalg.norm(place1.position - place2.position)
            cost = distance * (1 + self.random.random() * 0.1)
            self.costs[place1.id_no, place2.id_no] = cost
            self.costs[place2.id_no, place1.id_no] = cost
            place1.neighbors.add(place2.id_no)
            place2.neighbors.add(place1.id_no)

    def _generate_places(self):
        self.places = []  # type: List[Place]
        self.costs = np.ones(shape=(self.num_places, self.num_places)) * np.inf

        for place_idx in range(self.num_places):
            if place_idx == 0:
                place_position = self.position
            else:
                place_position = self.position + \
                    self.random.random(size=(2,)) * self.city_size
            place = Place(self.id_no, place_idx, place_position)
            self.costs[place_idx, place_idx] = 0
            if place_idx > 0:
                self.connect(place, self.places[-1])
            if place_idx == self.num_places - 1:
                self.connect(place, self.places[0])
            self.places.append(place)

        for place in self.places:
            nearest_np_places = sorted(
                self.places, key=lambda p: np.linalg.norm(place.position - p.position))[1: self.n_p + 1]
            for nearby_place in nearest_np_places:
                self.connect(place, nearby_place)

    def __repr__(self) -> str:
        return f'{self.id_no, self.position}'

    def __str__(self) -> str:
        return f'{self.id_no}'


class CityNavigation(SearchProblem):
    '''TODO: Description of the problem goes here'''

    def __init__(self, n_c, n_p, num_cities=150, num_places=150, state_size=100, city_size=1):
        # TODO: Feel free to add any args or kwargs. Can also do `*args, **kwargs`. These will be passed from MFMR.monitors.__init__.py
        # TODO: After this, fill out line 170 onwards in MFMR.monitors.__init__.py
        super().__init__()  # ! do not remove call to super class. See search_problem.py
        # self.cache = {}  # To cache heuristic value of states
        self.num_cities = num_cities
        self.num_places = num_places
        self.n_c = n_c
        self.n_p = n_p
        self.state_size = state_size
        self.city_size = city_size
        self.cities = []  # type: List[City]
        self.costs = np.ones(shape=(self.num_cities, self.num_cities)) * np.inf

    def seed(self, seed):
        """Set seeds of any random variables here.
        A random variable `self.random` is already provided by the super class, which is an instance of `numpy.random.RandomState`.
        It is preferred that any random stuff be done like `self.random.blablabla`. Its API is same as `numpy.random.blablabla`.
        """
        super().seed(seed)  # ! Do not remove this line. It seeds `self.random`

    def get_obs(self):
        '''returns problem instance specific arguments in a python list.
        ! Each argument is normalized to be between 0 a 1..
        '''
        return []

    def connect(self, city1: City, city2: City):
        if city2.id_no not in city1.neighbors:  # don't allow recalculation of cost
            distance = np.linalg.norm(city1.position - city2.position)
            cost = distance + 2
            self.costs[city1.id_no, city2.id_no] = cost
            self.costs[city2.id_no, city1.id_no] = cost
            city1.neighbors.add(city2.id_no)
            city2.neighbors.add(city1.id_no)

    def distance(self, location1, location2):
        city1, place1 = self.get_city_and_place(location1)
        city2, place2 = self.get_city_and_place(location2)
        return np.linalg.norm(place1.position - place2.position)

    def get_city_and_place(self, location: Tuple[int, int]):
        city = self.cities[location[0]]
        place = city.places[location[1]]
        return city, place

    def reset(self):
        '''Reset the search problem i.e. create a fresh instance.'''
        self.costs = np.ones(
            shape=(self.num_cities, self.num_cities)) * np.inf
        self.cities = []  # type: List[City]

        for city_idx in range(self.num_cities):
            city_position = self.random.random(size=(2,)) * self.state_size
            city = City(city_idx, city_position, self.n_p,
                        self.random, self.num_places, self.city_size)
            self.costs[city_idx, city_idx] = 0
            if city_idx > 0:
                self.connect(city, self.cities[-1])
            if city_idx == self.num_cities - 1:
                self.connect(city, self.cities[0])
            self.cities.append(city)

        for city in self.cities:
            nearest_nc_cities = sorted(
                self.cities, key=lambda c: np.linalg.norm(city.position - c.position))[1: self.n_c + 1]
            for nearby_city in nearest_nc_cities:
                self.connect(city, nearby_city)

        self.start_location, self.end_location = (0, 0)
        while self.start_location == self.end_location:
            self.start_location = (self.random.randint(
                self.num_cities), self.random.randint(self.num_places))
            self.end_location = (self.random.randint(
                self.num_cities), self.random.randint(self.num_places))

        self.start_state = self.start_location
        start_end_distance = self.distance(
            self.start_location, self.end_location)

        # TODO: After initialization. Populate self.info dictionary:
        self.info = {
            'starting_h': start_end_distance,
            'n_c': self.n_c,
            'n_p': self.n_p,
            'num_cities': self.num_cities,
            'num_places': self.num_places,
            'start_loc': self.start_location,
            'end_loc': self.end_location
        }
        # TODO: Log some interesting info about this particular instance:
        logger.info(
            f'Generated a city nav scenario. Start loc = {self.start_location}, end_loc = {self.end_location}')
        return self.start_state

    def get_children_nodes(self, parent_node):
        '''Gets children "Nodes" from parent "node". `Node` class is described in search_problem.py.
        '''
        # ! It is preffered not to do anything here. `self.successors(state)` method should be implemented instead, which returns successor "states".
        # ! The superclass handles everything there is to do with "nodes".
        # ! Leave this line as such if implementing self.successors:
        return super().get_children_nodes(parent_node)

    def goal_test(self, state):
        return state == self.end_location

    def successors(self, state: Tuple[int, int]):
        """Returns an iterable. Each item a dicationary of action and successor state."""
        # ! For good performance, especially for low memory consumption,
        # ! it is preferred to use `yield` style generator.

        cur_city, cur_place = self.get_city_and_place(state)

        for neighbor_place_id in cur_place.neighbors:
            neighbor_place = cur_city.places[neighbor_place_id]
            action = f'goto place {neighbor_place}'
            next_state = (cur_city.id_no, neighbor_place.id_no)
            yield {'state': next_state, 'action': action}

        if cur_place.id_no == 0:
            '''can go to other cities as well'''
            for neighbor_city_id in cur_city.neighbors:
                neighbor_city = self.cities[neighbor_city_id]
                action = f'goto city {neighbor_city}'
                next_state = (neighbor_city.id_no, 0)
                yield {'state': next_state, 'action': action}

    def cost(self, state, action, next_state):
        city1, place1 = self.get_city_and_place(state)
        city2, place2 = self.get_city_and_place(next_state)
        if city1 == city2:
            return city1.costs[place1.id_no, place2.id_no]
        else:
            return self.costs[city1.id_no, city2.id_no]

    def heuristic(self, state):
        return self.distance(state, self.end_location)

    def hash_state(self, state):
        '''provides a hash value which uniquely identifies the states'''
        return str(state)


if __name__ == "__main__":
    city_nav = CityNavigation(3, 3, 150, 150)
    city_nav.seed(0)
    state = city_nav.reset()
    print(city_nav.heuristic(state))
    nexts = city_nav.successors(state)
    for succ in nexts:
        a, ns = succ['action'], succ['state']
        cost = city_nav.cost(state, a, ns)
        print(state, a, ns)
        print(city_nav.heuristic(ns))
