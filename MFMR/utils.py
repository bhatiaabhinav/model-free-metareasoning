import json
import math


def update_mean(mean, n, x, add_x=True):
    """Calculates the updated mean of a collection on adding or removing an element. Returns 0 mean if updated number of elements become 0.

    Args:
        mean (float): current mean of the collection
        n (int): current number of elements in the collection
        x (float): the element to be either added or removed
        add_x (bool, optional): Whether to add (True) or remove the element. Defaults to True.

    Returns:
        float, int: updated mean, updated number of elements
    """
    mask = 2 * int(add_x) - 1
    new_n = n + mask
    if new_n == 0:
        return 0.0, 0
    if new_n < 0:
        raise ValueError('Cannot remove an element from an empty collection')
    sum_x = mean * n
    new_sum_x = sum_x + mask * x
    new_mean = new_sum_x / new_n
    return new_mean, new_n


def update_mean_std(mean, std, n, x, add_x=True):
    """Calculates the updated mean and std of a collection on adding or removing an element. Return 0 mean and std if updated number of elements become 0.

    Args:
        mean (float): current mean
        std (float): current std
        n (int): current number of elements in the collection
        x (float): the element to be either added or removed
        add_x (bool, optional): Whether to add (True) or remove the element. Defaults to True.

    Returns:
        float, int: updated std, updated number of elements
    """
    '''We will use the property that
        variance = E[(X-u)^2]
        = E[X^2 + u^2 - 2uX]
        = E[X^2] + u^2 - 2uE[X]
        = E[X^2] + u^2 -2u^2
        = E[X^2] - u^2
        = E[X^2] - (E[X])^2
    '''
    mask = 2 * add_x - 1
    new_n = n + mask
    if new_n == 0:
        return 0.0, 0.0, 0
    if new_n < 0:
        raise ValueError('Cannot remove an element from an empty collection')
    variance = std ** 2
    mean_x_squared = variance + mean ** 2  # by E[X^2] = variance + (E[X])^2
    '''now calculate new mean of X'''
    sum_x = mean * n
    new_sum_x = sum_x + mask * x
    new_mean = new_sum_x / new_n
    '''now calulate new mean of X^2'''
    sum_x_squared = mean_x_squared * n
    new_sum_x_squared = sum_x_squared + mask * x**2
    new_mean_x_squared = new_sum_x_squared / new_n
    '''now new std'''
    new_variance = new_mean_x_squared - new_mean**2
    assert new_variance > -0.1, "too much precision error?"
    new_variance = max(0, new_variance)  # a defence againt neg variance
    new_std = math.sqrt(new_variance)
    return new_mean, new_std, new_n


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


def get_csv_dictionary(file_path):
    with open(file_path) as file:
        entries = [line.strip().split(",") for line in file.readlines()]
        return {entry[0].lower(): float(entry[1]) for entry in entries}
