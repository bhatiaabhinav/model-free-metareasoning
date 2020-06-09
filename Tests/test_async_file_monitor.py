import random

import gym
import numpy as np
import pytest

import MFMR.monitors  # noqa
from MFMR.monitors.async_algo_monitor import AsyncAlgoMonitor

env_name = 'FileAlgoTest-v0'


def test_make_env():
    gym.make('FileAlgoTest-v0')


@pytest.fixture
def env():
    return gym.make('FileAlgoTest-v0')  # type: AsyncAlgoMonitor


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_basic(env, seed):
    random.seed(seed)
    env.seed(seed)
    hyperparams = np.array([])
    obs = env.reset()
    obs, r, d, i = env.step((env.CONTINUE_ACTION, hyperparams))
    print(obs, r, d, i)
    obs, r, d, i = env.step((env.CONTINUE_ACTION, hyperparams))
    print(obs, r, d, i)
    obs, r, d, i = env.step((env.CONTINUE_ACTION, hyperparams))
    print(obs, r, d, i)
    assert not i['interrupted']
    was_already_done = d
    obs, r, d, i = env.step((env.STOP_ACTION, hyperparams))
    print(obs, r, d, i)
    assert d
    if not was_already_done:
        assert i['interrupted']


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_final_solution(env, seed):
    random.seed(seed)
    env.seed(seed)
    hyperparams = np.array([])
    obs = env.reset()
    d = False
    prev_sol_qual = -np.inf
    while not d:
        obs, r, d, i = env.step((env.CONTINUE_ACTION, hyperparams))
        assert not i['interrupted']
        assert i['solution_quality'] >= prev_sol_qual
        prev_sol_qual = i['solution_quality']
    assert not i['interrupted']


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_interrupted(env, seed):
    random.seed(seed)
    env.seed(seed)
    hyperparams = np.array([])
    obs = env.reset()
    obs, r, d, i = env.step((env.STOP_ACTION, hyperparams))
    assert d
    assert i['interrupted']


# in command line, run `pytest Tests/` after activating virtualenv
