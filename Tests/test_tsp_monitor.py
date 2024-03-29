import random

import gym
import numpy as np
import pytest

import MFMR.monitors  # noqa
from MFMR.monitors.async_algo_monitor import AsyncAlgoMonitor

env_name = 'Tsp30-v0'


def test_make_env():
    gym.make(env_name)


@pytest.fixture
def env():
    return gym.make(env_name)  # type: AsyncAlgoMonitor


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_basic(env, seed):
    random.seed(seed)
    env.seed(seed)
    obs = env.reset()
    obs, r, d, i = env.step(env.CONTINUE_ACTION)
    print(obs, r, d, i)
    obs, r, d, i = env.step(env.CONTINUE_ACTION)
    print(obs, r, d, i)
    obs, r, d, i = env.step(env.CONTINUE_ACTION)
    print(obs, r, d, i)
    assert not i['interrupted']
    if d:
        assert i['graceful_exit']
    else:
        obs, r, d, i = env.step(env.STOP_ACTION)
        print(obs, r, d, i)
        assert d
        if not i['graceful_exit']:
            assert i['interrupted']


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_final_solution(env, seed):
    random.seed(seed)
    env.seed(seed)
    obs = env.reset()
    d = False
    prev_sol_qual = -np.inf
    while not d:
        obs, r, d, i = env.step(env.CONTINUE_ACTION)
        assert not i['interrupted']
        assert i['solution_quality'] >= prev_sol_qual
        prev_sol_qual = i['solution_quality']
    assert d
    assert not i['interrupted']
    assert i['graceful_exit']


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_interrupted(env, seed):
    random.seed(seed)
    env.seed(seed)
    obs = env.reset()
    obs, r, d, i = env.step(env.STOP_ACTION)
    print(obs, r, d, i)
    assert d
    if not i['graceful_exit']:
        assert i['interrupted']


# in command line, run `pytest Tests/` after activating virtualenv
