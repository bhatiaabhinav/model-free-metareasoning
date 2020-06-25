import random

import numpy as np
from scipy.spatial import distance

from utils import get_key

BLANK_SYMBOL = 0
ACTIONS = {
    'North': (-1, 0),
    'East': (0, -1),
    'South': (1, 0),
    'West': (0, 1)
}
HEURISTIC_CACHE = {}


def get_initial_puzzle(size):
    puzzle = np.zeros((size, size))

    for row in range(size):
        for column in range(size):
            puzzle[row, column] = (row * size) + column + 1

    puzzle[size - 1, size - 1] = BLANK_SYMBOL

    return puzzle


def get_next_puzzle(puzzle, action):
    location = get_blank_location(puzzle)

    x, y = location
    next_x, next_y = get_next_blank_location(location, action)

    next_state = np.copy(puzzle)
    next_state[x, y], next_state[next_x, next_y] = next_state[next_x, next_y],  next_state[x, y]

    return next_state


def get_size(puzzle):
    return puzzle.shape[0]


def get_blank_location(puzzle):
    locations = np.where(puzzle == BLANK_SYMBOL)
    x, y = locations[0][0], locations[1][0]
    return x, y


def get_next_blank_location(location, action):
    x, y = location
    delta_x, delta_y = ACTIONS[action]
    new_x, new_y = x + delta_x, y + delta_y
    return new_x, new_y


def is_valid_blank_location(puzzle, location):
    size = get_size(puzzle)
    x, y = location
    return size > x >= 0 and size > y >= 0


def get_random_puzzle(size, iterations=100):
    puzzle = get_initial_puzzle(size)
    actions = list(ACTIONS.keys())

    for _ in range(iterations):
        location = get_blank_location(puzzle)
        action = random.choice(actions)
        next_location = get_next_blank_location(location, action)

        while not is_valid_blank_location(puzzle, next_location):
            action = random.choice(actions)
            next_location = get_next_blank_location(location, action)

        puzzle = get_next_puzzle(puzzle, action)

    return puzzle


def get_difficult_puzzle(size, target_difficulty, epsilon=0.1):
    puzzle = get_initial_puzzle(size)
    actions = list(ACTIONS.keys())

    difficulty = get_manhattan_distance(puzzle)
    while difficulty < target_difficulty:
        action = random.choice(actions)
        location = get_blank_location(puzzle)
        next_location = get_next_blank_location(location, action)

        if is_valid_blank_location(puzzle, next_location):
            next_puzzle = get_next_puzzle(puzzle, action)
            next_difficulty = get_manhattan_distance(next_puzzle)

            if next_difficulty > difficulty or random.random() < epsilon:
                puzzle = next_puzzle
                difficulty = next_difficulty

    return puzzle


def get_manhattan_distance(puzzle):
    puzzle_key = get_key(puzzle)
    if puzzle_key in HEURISTIC_CACHE:
        return HEURISTIC_CACHE[puzzle_key]

    size = get_size(puzzle)
    goal_puzzle = get_initial_puzzle(size)
    manhattan_distance = 0

    for row in range(size):
        for column in range(size):
            value = goal_puzzle[row, column]

            if value == BLANK_SYMBOL:
                continue

            locations = np.where(puzzle == value)
            x, y = locations[0][0], locations[1][0]
            manhattan_distance += distance.cityblock((x, y), (row, column))

    HEURISTIC_CACHE[puzzle_key] = manhattan_distance

    return manhattan_distance


def is_goal(state):
    size = get_size(state)
    goal = get_initial_puzzle(size)
    return np.array_equal(state, goal)


def get_successors(state):
    successors = []

    for action in ACTIONS:
        location = get_blank_location(state)
        next_location = get_next_blank_location(location, action)

        if is_valid_blank_location(state, next_location):
            next_puzzle = get_next_puzzle(state, action)
            successors.append({'state': next_puzzle, 'action': action})

    return successors


def get_cost(state, action, next_state):
    return 1


def get_heuristic(state):
    return get_manhattan_distance(state)
