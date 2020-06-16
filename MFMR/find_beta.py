import random

import gym
import matplotlib.pyplot as plt

import MFMR.monitors  # noqa
from MFMR.monitors.async_algo_monitor import AsyncAlgoMonitor

random.seed(0)
metareasoning_env = gym.make('Tsp30-Cont-v0')  # type: AsyncAlgoMonitor

metareasoning_env.reset()
quality = metareasoning_env.algo.get_solution_quality()
qualities = [quality]

utility = metareasoning_env.get_cur_utility()
utilities = [utility]

steps = 0

while True:
    obs, r, is_episode_done, info = metareasoning_env.step(
        metareasoning_env.CONTINUE_ACTION)

    steps += 1
    time = metareasoning_env.get_time()
    quality = metareasoning_env.get_solution_quality()
    qualities.append(quality)
    utility = metareasoning_env.get_cur_utility()
    utilities.append(utility)

    print(steps, time, quality, utility)

    if is_episode_done:
        break

plt.figure(figsize=(7, 3))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["grid.linestyle"] = "-"
plt.xlabel("Steps")
plt.ylabel("Utilities")
plt.grid(True)

axis = plt.gca()
axis.spines["top"].set_visible(False)
# axis.set_xlim([0, 2 * utilities.index(max(utilities))])
# axis.set_ylim([utilities[0], 1.05 * max(utilities)])

plt.plot(range(len(utilities)), utilities, color="r")
plt.tight_layout()
plt.show()
