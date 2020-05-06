import json
import math

import numpy as np
import scipy.stats as stats


def save(filename, data):
    with open(filename, "w") as file:
        json.dump(data, file)


def load(filename):
    with open(filename, "r") as file:
        return json.load(file)


def save_policy(policy, filename):
    with open(filename, 'w') as file:
        json.dump(policy, file, sort_keys=True, indent=4)


def digitize(item, bins):
    for i, _ in enumerate(bins):
        if i + 1 < len(bins):
            if bins[i] <= item < bins[i + 1]:
                return i
    return len(bins) - 1


def get_dataset(problem_file, increment, transformer):
    instances = load(problem_file)

    dataset = []
    for instance in instances.values():
        entries = list(enumerate(instance["estimated_qualities"]))

        if transformer:
            dataset.append([(transformer(quality), round(time / increment)) for time, quality in entries[0:len(entries):increment]])
        else:
            dataset.append([(quality, round(time / increment)) for time, quality in entries[0:len(entries):increment]])

    return dataset


def get_time_dependent_utility(quality, time, alpha, beta):
    return alpha * quality - math.exp(beta * time)


def get_convergence_point(data, threshold, period):
    threshold_iterations = 0

    for i in range(len(data) - 1):
        difference = abs(data[i] - data[i - 1])

        if difference <= threshold:
            threshold_iterations += 1
            if threshold_iterations >= period:
                return {"episode": i, "error": data[i]}
        else:
            threshold_iterations = 0


def get_smoothed_values(data, window_size):
    smoothed_values = []

    for i in range(1, len(data)):
        truncated_data = data[:i]
        windowed_data = truncated_data[-window_size:]
        smoothed_values.append(np.average(windowed_data))

    return smoothed_values


def get_results(errors, window_size, plot_window_size):
    return {
        "mean_error": np.average(errors[-window_size:]),
        "standard_deviation_error": stats.sem(errors[-window_size:]),
        "smoothed_errors": get_smoothed_values(errors, plot_window_size)
    }
