import logging

import numpy as np
from wandb.apis.public import Run
from MFMR.algos.search_problem import SearchProblem
from scipy.spatial import distance

logger = logging.getLogger(__name__)
ldebug = logger.isEnabledFor(logging.DEBUG)


BLANK_SYMBOL = 0
ACTIONS = {
    'North': (-1, 0),
    'East': (0, 1),
    'South': (1, 0),
    'West': (0, -1)
}


class NPuzzle(SearchProblem):
    def __init__(self, N_range, inverse=False):
        super().__init__()
        self.N_range = N_range
        self.inverse = inverse
        self.N = self.N_range[0] + \
            self.random.randint(self.N_range[1] + 1 - self.N_range[0])
        self.start_state = None
        self.cur_difficulty = None
        self.cur_relative_difficulty = None
        self.cache = {}
        self.cache_inverse = {}

    def compute_max_difficulty(self, n):
        # 4n (n-1 + n-3 + n-5 + ...)
        s = 0
        k = 1
        while n - k >= 0:
            s += n - k
            k += 2
        return 4 * n * s

    def seed(self, seed):
        super().seed(seed)

    def get_obs(self):
        return [self.N / self.N_range[1], self.cur_relative_difficulty]

    def reset(self):
        self.cache.clear()  # TODO: Should work withouot this
        self.cache_inverse.clear()
        self.N = self.N_range[0] + \
            self.random.randint(self.N_range[1] + 1 - self.N_range[0])
        self.initial_puzzle = self.get_initial_puzzle()
        self.max_difficulty = self.compute_max_difficulty(self.N)
        min_difficulty = 30
        max_difficulty = 40
        self.cur_difficulty = min_difficulty + \
            self.random.randint(max_difficulty + 1 - min_difficulty)
        self.cur_relative_difficulty = self.cur_difficulty / self.max_difficulty
        # print(self.N, self.cur_difficulty, self.max_difficulty)
        # raise RuntimeError()
        self.start_state = self.get_difficult_puzzle(self.cur_difficulty)
        self.info = {
            'puzzle_n': self.N,
            'puzzle_difficulty': self.cur_relative_difficulty,
            # 'puzzle': self.start_state
        }
        logger.info(
            f'puzzle={self.start_state}, puzzle_n={self.N}, difficulty={self.cur_relative_difficulty}')

    def get_initial_puzzle(self):
        size = self.N
        puzzle = np.zeros((size, size))

        for row in range(size):
            for column in range(size):
                puzzle[row, column] = (row * size) + column + 1

        puzzle[size - 1, size - 1] = BLANK_SYMBOL

        return puzzle

    def get_next_puzzle(self, puzzle, action):
        location = self.get_blank_location(puzzle)

        x, y = location
        next_x, next_y = self.get_next_blank_location(location, action)

        next_state = np.copy(puzzle)
        next_state[x, y], next_state[next_x,
                                     next_y] = next_state[next_x, next_y], next_state[x, y]

        return next_state

    def get_blank_location(self, puzzle):
        locations = np.where(puzzle == BLANK_SYMBOL)
        x, y = locations[0][0], locations[1][0]
        return x, y

    def get_next_blank_location(self, location, action):
        x, y = location
        delta_x, delta_y = ACTIONS[action]
        new_x, new_y = x + delta_x, y + delta_y
        return new_x, new_y

    def is_valid_blank_location(self, puzzle, location):
        x, y = location
        return self.N > x >= 0 and self.N > y >= 0

    def get_random_puzzle(self, iterations=100):
        puzzle = self.initial_puzzle
        actions = list(ACTIONS.keys())

        for _ in range(iterations):
            location = self.get_blank_location(puzzle)
            action = self.random.choice(actions)
            next_location = self.get_next_blank_location(location, action)

            while not self.is_valid_blank_location(puzzle, next_location):
                action = self.random.choice(actions)
                next_location = self.get_next_blank_location(location, action)

            puzzle = self.get_next_puzzle(puzzle, action)

        return puzzle

    def get_difficult_puzzle(self, target_difficulty, epsilon=0.1):
        puzzle = self.initial_puzzle
        # print('puzzle', puzzle)
        actions = list(ACTIONS.keys())

        difficulty = self.get_manhattan_distance(puzzle)
        # print('diff', difficulty)
        while difficulty < target_difficulty:
            action = self.random.choice(actions)
            # print('action', action)
            location = self.get_blank_location(puzzle)
            # print('blank loc', location)
            next_location = self.get_next_blank_location(location, action)
            # print('next blank loc', next_location)

            if self.is_valid_blank_location(puzzle, next_location):
                # print('valid loc')
                next_puzzle = self.get_next_puzzle(puzzle, action)
                next_difficulty = self.get_manhattan_distance(next_puzzle)
                # print('next puzzle', next_puzzle, 'next diff', next_difficulty)

                if next_difficulty > difficulty or self.random.random() < epsilon:
                    # print('using this')
                    puzzle = next_puzzle
                    difficulty = next_difficulty
                else:
                    # print('discarding because diff decreased')
                    pass
            else:
                # print('not valid loc')
                pass

        return puzzle

    def get_manhattan_distance(self, puzzle, inverse=False):
        puzzle_key = self.hash_state(puzzle)
        if inverse:
            if puzzle_key in self.cache_inverse:
                return self.cache_inverse[puzzle_key]
        else:
            if puzzle_key in self.cache:
                return self.cache[puzzle_key]

        size = self.N
        goal_puzzle = self.initial_puzzle
        manhattan_distance = 0

        for row in range(size):
            for column in range(size):
                value = goal_puzzle[row, column]

                if value == BLANK_SYMBOL:
                    continue

                locations = np.where(puzzle == value)
                x, y = locations[0][0], locations[1][0]
                if inverse:
                    manhattan_distance += distance.cityblock(
                        (x, y), (row, column)) / value
                else:
                    manhattan_distance += distance.cityblock(
                        (x, y), (row, column))

        if inverse:
            self.cache_inverse[puzzle_key] = manhattan_distance
        else:
            self.cache[puzzle_key] = manhattan_distance

        return manhattan_distance

    def get_children_nodes(self, parent):
        return super().get_children_nodes(parent)

    def goal_test(self, state):
        goal = self.initial_puzzle
        return np.array_equal(state, goal)

    def successors(self, state):
        for action in ACTIONS:
            location = self.get_blank_location(state)
            next_location = self.get_next_blank_location(location, action)

            if self.is_valid_blank_location(state, next_location):
                next_puzzle = self.get_next_puzzle(state, action)
                successor = {'state': next_puzzle, 'action': action}
                yield successor

    def cost(self, state, action, next_state):
        if self.inverse:
            moved_tile = np.sum(
                np.abs(np.asarray(state) - np.asarray(next_state))) / 2
            # print(state, action, next_state, moved_tile)
            return 1 / moved_tile
        return 1

    def heuristic(self, state):
        return self.get_manhattan_distance(state, inverse=self.inverse)

    def hash_state(self, state):
        return str(state.tolist())
