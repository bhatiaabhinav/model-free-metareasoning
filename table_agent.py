from __future__ import division

import random

import numpy as np


class Agent:
    def __init__(self, params, env, action_value_function=None):
        self.params = params
        self.env = env
        self.action_value_function = action_value_function if action_value_function else {state: [random.random(), random.random()] for state in env.get_states()}

    def get_optimal_action(self, state):
        return np.argmax(self.action_value_function[state])

    def get_action(self, state):
        if random.random() > self.params["epsilon"]:
            return self.get_optimal_action(state)
        return random.choice(self.env.ACTIONS)

    def run_q_learning(self, statistics):
        print("Running tabular Q-learning with the parameters {}".format(self.params))

        for _ in range(self.params["episodes"]):
            state = self.env.reset()
            action = self.get_action(state)

            while True:
                next_state, reward, is_episode_done = self.env.step(action)

                current_value = self.action_value_function[state][action]
                estimated_value = reward + self.params["gamma"] * max(self.action_value_function[next_state])
                value_error = estimated_value - current_value
                self.action_value_function[state][action] += self.params["alpha"] * value_error

                next_action = self.get_action(state)

                if is_episode_done:
                    utility = self.env.get_utility()
                    optimal_utility = self.env.get_optimal_utility()
                    error = abs((utility - optimal_utility) / optimal_utility)

                    statistics["errors"].append(error)
                    statistics["stopping_points"].append(next_state[1])
                    statistics["utilities"].append(utility)

                    self.params["epsilon"] *= self.params["decay"]

                    break

                state = next_state
                action = next_action

    def run_sarsa(self, statistics):
        print("Running tabular SARSA with the parameters {}".format(self.params))

        for _ in range(self.params["episodes"]):
            state = self.env.reset()
            action = self.get_action(state)

            while True:
                next_state, reward, is_episode_done = self.env.step(action)

                current_value = self.action_value_function[state][action]
                estimated_value = reward + self.params["gamma"] * self.action_value_function[next_state][action]
                value_error = estimated_value - current_value
                self.action_value_function[state][action] += self.params["alpha"] * value_error

                next_action = self.get_action(state)

                if is_episode_done:
                    utility = self.env.get_utility()
                    optimal_utility = self.env.get_optimal_utility()
                    error = abs((utility - optimal_utility) / optimal_utility)

                    statistics["errors"].append(error)
                    statistics["stopping_points"].append(next_state[1])
                    statistics["utilities"].append(utility)

                    self.params["epsilon"] *= self.params["decay"]

                    break

                state = next_state
                action = next_action
