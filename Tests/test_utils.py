import numpy as np
import pytest

from MFMR.utils import update_mean, update_mean_std, update_mean_std_corr


@pytest.mark.parametrize("seed", range(20))
def test_update_mean_std_on_add(seed):
    random = np.random.RandomState(seed)
    n = random.randint(2, 11)
    numbers = list(10000000 * (2 * random.random(n) - 1))
    x = 10000000 * random.randn()
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


@pytest.mark.parametrize("seed", range(20))
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

    u_mean, u_std, u_n = update_mean_std(mean, std, n, x, remove=True)
    assert abs(u_mean - mean_new) <= 1e-6 * abs(mean_new)
    assert abs(u_std - std_new) <= 1e-6 * abs(mean_new)
    assert u_n == n - 1

    u_mean, u_n = update_mean(mean, n, x, remove=True)
    assert abs(u_mean - mean_new) <= 1e-6 * abs(mean_new)
    assert u_n == n - 1


@pytest.mark.parametrize("seed", range(20))
def test_update_mean_std_corr_on_add(seed):
    random = np.random.RandomState(seed)
    n = random.randint(2, 11)
    numbers_x = list(10000000 * (2 * random.random(n) - 1))
    numbers_y = list(10000000 * (2 * random.random(n) - 1))
    x = 10000000 * random.randn()
    y = 10000000 * random.randn()
    numbers_new_x = numbers_x + [x]
    numbers_new_y = numbers_y + [y]
    mean_x = np.mean(numbers_x)
    std_x = np.std(numbers_x)
    mean_y = np.mean(numbers_y)
    std_y = np.std(numbers_y)
    corr = np.corrcoef(numbers_x, numbers_y)[0, 1]
    mean_new_x = np.mean(numbers_new_x)
    std_new_x = np.std(numbers_new_x)
    mean_new_y = np.mean(numbers_new_y)
    std_new_y = np.std(numbers_new_y)
    corr_new = np.corrcoef(numbers_new_x, numbers_new_y)[0, 1]

    u_mean_x, u_mean_y, u_std_x, u_std_y, u_corr, u_n = update_mean_std_corr(
        mean_x, mean_y, std_x, std_y, corr, n, x, y)

    assert abs(u_mean_x - mean_new_x) <= 1e-6 * abs(mean_new_x)
    assert abs(u_std_x - std_new_x) <= 1e-6 * abs(mean_new_x)
    assert abs(u_mean_y - mean_new_y) <= 1e-6 * abs(mean_new_y)
    assert abs(u_std_y - std_new_y) <= 1e-6 * abs(mean_new_y)
    assert abs(u_corr - corr_new) <= 1e-6
    assert u_n == n + 1
