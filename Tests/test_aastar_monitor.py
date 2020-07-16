import random

import gym
import numpy as np
import pytest
from gym.wrappers import TimeLimit

import MFMR.monitors  # noqa
from MFMR.monitors.async_algo_monitor import AsyncAlgoMonitor

env_name = 'A1.5Astar-4puzzle-medium-B0.3-v0'


def test_make_env():
    env = gym.make(env_name)
    env = TimeLimit(env, max_episode_steps=150)
    return env


@pytest.fixture
def env():
    env = gym.make(env_name)
    env = TimeLimit(env, max_episode_steps=150)  # type: AsyncAlgoMonitor
    return env


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
    was_already_done = d
    obs, r, d, i = env.step(env.STOP_ACTION)
    print(obs, r, d, i)
    assert d
    if not was_already_done:
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
    assert not i['interrupted']


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_interrupted(env, seed):
    random.seed(seed)
    env.seed(seed)
    obs = env.reset()
    obs, r, d, i = env.step(env.STOP_ACTION)
    assert d
    assert i['interrupted']


# in command line, run `pytest Tests/` after activating virtualenv
