import numpy as np
import pytest

from MFMR.utils import update_mean, update_mean_std


@pytest.mark.parametrize("seed", range(100))
def test_update_mean_std_on_add(seed):
    random = np.random.RandomState(seed)
    n = random.randint(2, 11)
    numbers = list(10000000 * (2 * random.random(n) - 1))
    x = random.randn()
    numbers_new = numbers + [x]
    mean = np.mean(numbers)
    std = np.std(numbers)
    mean_new = np.mean(numbers_new)
    std_new = np.std(numbers_new)

    u_mean, u_std, u_n = update_mean_std(mean, std, n, x)
    assert abs(u_mean - mean_new) <= 1e-6 * abs(mean_new)
    assert abs(u_std - std_new) <= 1e-6 * abs(mean_new)
    assert u_n == n + 1

    u_mean, u_n = update_mean(mean, n, x)
    assert abs(u_mean - mean_new) <= 1e-6 * abs(mean_new)
    assert u_n == n + 1


@pytest.mark.parametrize("seed", range(100))
def test_update_mean_std_on_substract(seed):
    random = np.random.RandomState(seed)
    n = random.randint(2, 11)
    numbers = list(10000000 * (2 * random.random(n) - 1))
    numbers_new = numbers[0:-1]
    x = numbers[-1]
    mean = np.mean(numbers)
    std = np.std(numbers)
    mean_new = np.mean(numbers_new)
    std_new = np.std(numbers_new)

    u_mean, u_std, u_n = update_mean_std(mean, std, n, x, add_x=False)
    assert abs(u_mean - mean_new) <= 1e-6 * abs(mean_new)
    assert abs(u_std - std_new) <= 1e-6 * abs(mean_new)
    assert u_n == n - 1

    u_mean, u_n = update_mean(mean, n, x, add_x=False)
    assert abs(u_mean - mean_new) <= 1e-6 * abs(mean_new)
    assert u_n == n - 1
