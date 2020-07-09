import numpy as np
from scipy.spatial import distance

from MFMR.algos.search_problem import SearchProblem, get_key

BLANK_SYMBOL = 0
ACTIONS = {
    'North': (-1, 0),
    'East': (0, -1),
    'South': (1, 0),
    'West': (0, 1)
}


class NPuzzle(SearchProblem):
    def __init__(self, n, difficulty_string):
        super().__init__()
        self.n = n
        self.difficulty_string = difficulty_string
        self.initial_puzzle = self.get_initial_puzzle()
        self.max_difficulty = self.compute_max_difficulty()
        print('Max diff is', self.max_difficulty)
        self.difficulty_map = {
            'easy': self.max_difficulty // 4,
            'medium': self.max_difficulty // 2,
            'hard': self.max_difficulty - 4
        }
        self.difficulty = self.difficulty_map[self.difficulty_string]
        print('Diff is', self.difficulty)
        self.cache = {}

    def compute_max_difficulty(self):
        # 4n (n-1 + n-3 + n-5 + ...)
        s = 0
        k = 1
        while self.n - k >= 0:
            s += self.n - k
            k += 2
        return 4 * self.n * s

    def reset(self):
        self.start_state = self.get_difficult_puzzle(self.difficulty)
        self.cache.clear()

    def get_initial_puzzle(self):
        size = self.n
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
        return self.n > x >= 0 and self.n > y >= 0

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
        actions = list(ACTIONS.keys())

        difficulty = self.get_manhattan_distance(puzzle)
        while difficulty < target_difficulty:
            action = self.random.choice(actions)
            location = self.get_blank_location(puzzle)
            next_location = self.get_next_blank_location(location, action)

            if self.is_valid_blank_location(puzzle, next_location):
                next_puzzle = self.get_next_puzzle(puzzle, action)
                next_difficulty = self.get_manhattan_distance(next_puzzle)

                if next_difficulty > difficulty or self.random.random() < epsilon:
                    puzzle = next_puzzle
                    difficulty = next_difficulty

        return puzzle

    def get_manhattan_distance(self, puzzle):
        puzzle_key = get_key(puzzle)
        if puzzle_key in self.cache:
            return self.cache[puzzle_key]

        size = self.n
        goal_puzzle = self.initial_puzzle
        manhattan_distance = 0

        for row in range(size):
            for column in range(size):
                value = goal_puzzle[row, column]

                if value == BLANK_SYMBOL:
                    continue

                locations = np.where(puzzle == value)
                x, y = locations[0][0], locations[1][0]
                manhattan_distance += distance.cityblock((x, y), (row, column))

        self.cache[puzzle_key] = manhattan_distance

        return manhattan_distance

    def goal_test(self, state):
        goal = self.initial_puzzle
        return np.array_equal(state, goal)

    def successors(self, state):
        successors = []

        for action in ACTIONS:
            location = self.get_blank_location(state)
            next_location = self.get_next_blank_location(location, action)

            if self.is_valid_blank_location(state, next_location):
                next_puzzle = self.get_next_puzzle(state, action)
                successors.append({'state': next_puzzle, 'action': action})

        return successors

    def cost(self, state, action, next_state):
        return 1

    def heuristic(self, state):
        return self.get_manhattan_distance(state)
