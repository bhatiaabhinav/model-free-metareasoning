from __future__ import division

import random

import numpy as np

import utils
from function_approximator import FunctionApproximator


class Agent:
    def __init__(self, params, env, weights=None, action_value_function=None):
        self.params = params
        self.env = env

        if weights and action_value_function:
            self.function_approximator = FunctionApproximator(params, env, weights, action_value_function)
            self.action_value_function = action_value_function
        else:
            self.function_approximator = FunctionApproximator(params, env)
            self.action_value_function = self.function_approximator.get_initial_action_value_function()

    def get_optimal_action(self, state):
        return np.argmax(self.action_value_function[state])

    def get_action(self, state):
        if random.random() > self.params["epsilon"]:
            return self.get_optimal_action(state)
        return random.choice(self.env.ACTIONS)

    def get_policy(self):
        policy = {int(quality_class): [] for quality_class in self.env.QUALITY_CLASSES}

        for key in policy.keys():
            for time_class in self.env.TIME_CLASSES:
                policy[key].append(self.env.ACTION_MAP[self.get_optimal_action((key, time_class))])

        return policy

    def run_q_learning(self, statistics):
        print("Running Fourier Q-learning with the parameters {}".format(self.params))

        for episode in range(self.params["episodes"]):
            state = self.env.reset()
            psi = self.function_approximator.calculate_fourier_approximation(state)

            while True:
                self.action_value_function = self.function_approximator.update_action_value_function(state, psi)
                action = self.get_action(state)

                next_state, reward, is_episode_done = self.env.step(action)

                if is_episode_done:
                    utility = self.env.get_utility()
                    optimal_utility = self.env.get_optimal_utility()
                    error = abs((utility - optimal_utility) / optimal_utility)

                    statistics["errors"].append(error)
                    statistics["stopping_points"].append(next_state[1])
                    statistics["utilities"].append(utility)

                    if "checkpoint" in self.params and episode % self.params["checkpoint"] == 0:
                        utils.save_policy(self.get_policy(), "%s-checkpoint-policy.json" % episode)

                    self.params["epsilon"] *= self.params["decay"]

                    break

                next_action = self.get_optimal_action(next_state)
                next_psi = self.function_approximator.calculate_fourier_approximation(next_state)
                self.function_approximator.update_weights(action, next_action, psi, next_psi, reward)

                state = next_state
                psi = next_psi

    def run_sarsa(self, statistics):
        print("Running Fourier SARSA with the parameters {}".format(self.params))

        for episode in range(self.params["episodes"]):
            state = self.env.reset()
            psi = self.function_approximator.calculate_fourier_approximation(state)

            self.action_value_function = self.function_approximator.update_action_value_function(state, psi)
            action = self.get_action(state)

            while True:
                next_state, reward, is_episode_done = self.env.step(action)
                next_psi = self.function_approximator.calculate_fourier_approximation(next_state)

                self.action_value_function = self.function_approximator.update_action_value_function(next_state, next_psi)
                next_action = self.get_action(next_state)

                self.function_approximator.update_weights(action, next_action, psi, next_psi, reward)

                if is_episode_done:
                    utility = self.env.get_utility()
                    optimal_utility = self.env.get_optimal_utility()
                    error = abs((utility - optimal_utility) / optimal_utility)

                    statistics["errors"].append(error)
                    statistics["stopping_points"].append(next_state[1])
                    statistics["utilities"].append(utility)

                    if "checkpoint" in self.params and episode % self.params["checkpoint"] == 0:
                        utils.save_policy(self.get_policy(), "%s-checkpoint-policy.json", episode)

                    self.params["epsilon"] *= self.params["decay"]

                    break

                state = next_state
                psi = next_psi
                action = next_action
