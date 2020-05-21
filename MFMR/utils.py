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


def digitize(item, bins):
    for i, _ in enumerate(bins):
        if i + 1 < len(bins):
            if bins[i] <= item < bins[i + 1]:
                return i
    return len(bins) - 1


def get_dataset(problem_file, increment):
    instances = load(problem_file)

    dataset = []
    for instance in instances.values():
        entries = list(enumerate(instance["estimated_qualities"]))
        dataset.append([(quality, round(time / increment))
                        for time, quality in entries[0:len(entries):increment]])

    return dataset


def get_time_dependent_utility(quality, time, alpha, beta):
    return alpha * quality - math.exp(beta * time)
