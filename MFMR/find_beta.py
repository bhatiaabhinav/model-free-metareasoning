import os
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import TimeLimit

import MFMR.monitors  # noqa
from MFMR.monitors.async_algo_monitor import AsyncAlgoMonitor


def main():
    random.seed(0)
    metareasoning_env = gym.make(
        'A2.0Astar-20tsp-B0.3-v0')  # type: AsyncAlgoMonitor
    # metareasoning_env = TimeLimit(metareasoning_env, max_episode_steps=150)

    metareasoning_env.seed(0)
    metareasoning_env.reset()
    quality = metareasoning_env.algo.get_solution_quality()
    qualities = [quality]
    qualities_ub = [np.inf]
    times = [0]
    utility = metareasoning_env.get_cur_utility()
    utilities = [utility]

    steps = 0

    while True:
        obs, r, is_episode_done, info = metareasoning_env.step(0)
        # metareasoning_env.render()

        # print(obs)
        # print(info)

        steps += 1
        time = metareasoning_env.get_time()
        times.append(time)
        quality = metareasoning_env.get_solution_quality()
        qualities.append(quality)
        utility = metareasoning_env.get_cur_utility()
        utilities.append(utility)
        qualities_ub.append(info['q_ub'] * metareasoning_env.alpha)

        print('steps', steps, 't', time, 'q',
              quality, 'u', utility, 'w', info['w'], 'n', info['tsp_n'], 'b', info['beta'], 'q_ub', info['q_ub'])

        if is_episode_done:
            break

    metareasoning_env.close()

    plt.figure(figsize=(7, 3))
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["grid.linestyle"] = "-"
    plt.xlabel("Time")
    plt.ylabel("Utilities")
    plt.grid(True)

    axis = plt.gca()
    axis.spines["top"].set_visible(False)
    # axis.set_xlim([0, 2 * utilities.index(max(utilities))])
    # axis.set_ylim([utilities[0], 1.05 * max(utilities)])

    os.makedirs('output', exist_ok=True)
    plt.plot(times, utilities, color="r")
    plt.plot(times, qualities_ub, color="g")
    plt.tight_layout()
    plt.savefig('output/beta_plot.png')


if __name__ == "__main__":
    main()
