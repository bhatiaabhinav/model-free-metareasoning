import os
import random
import sys

import gym
import matplotlib.pyplot as plt
import numpy as np

import MFMR.monitors  # noqa
from MFMR.monitors.async_algo_monitor import AsyncAlgoMonitor


def main():
    random.seed(0)
    name = f'A{sys.argv[2]}Astar-4puzzle-v0'
    # name = f'A{sys.argv[2]}Astar-25tsp-v0'
    metareasoning_env = gym.make(name)  # type: AsyncAlgoMonitor

    metareasoning_env.seed(int(sys.argv[1]))
    metareasoning_env.reset()
    alpha = metareasoning_env.alpha
    scale = 10
    quality = metareasoning_env.algo.get_solution_quality() * scale
    qualities = [quality]
    qualities_ub = [np.inf]
    times = [0]
    utility = metareasoning_env.get_cur_utility() * scale / alpha
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
        quality = metareasoning_env.get_solution_quality() * scale
        qualities.append(quality)
        utility = metareasoning_env.get_cur_utility() * scale / alpha
        utilities.append(utility)
        q_ub = info['q_ub'] * scale
        qualities_ub.append(q_ub)

        if 'tsp' in name:
            print('steps', steps, 't', time, 'q', quality, 'u', utility, 'w',
                  info['w'], 'av_w', info['w_av'], 'n', info['tsp_n'], 'sparsity', info['tsp_sparsity'], 'b', info['beta'], 'q_ub', q_ub, 'cpu', info['cpu'])
        elif 'puzzle' in name:
            print('steps', steps, 't', time, 'q',
                  quality, 'u', utility, 'w', info['w'], 'av_w', info['w_av'], 'n', info['puzzle_n'], 'diff', info['puzzle_difficulty'], 'b', info['beta'], 'q_ub', q_ub, 'cpu', info['cpu'])

        if is_episode_done:
            break

    print('Max Utility', max(utilities))
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
    plt.savefig('output/example_episode.png')


if __name__ == "__main__":
    main()
