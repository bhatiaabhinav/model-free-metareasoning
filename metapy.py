import argparse
import sys

import matplotlib.pyplot as plt

import env
import fourier_agent
import table_agent
import utils

WINDOW_SIZE = 100

ALPHA = 200
BETA = 0.02
INCREMENT = 5


def main():
    parser = argparse.ArgumentParser(description="Run a reinforcement learning agent with a function approximation.")
    parser.add_argument("-method", type=str, required=True, help="The reinforcement learning agent")
    parser.add_argument("-problem", type=str, required=True, help="The problem file")
    parser.add_argument("-function", type=str, required=False, help="The function approximation")
    parser.add_argument("-order", type=int, required=False, help="The Fourier basis order")
    parser.add_argument("-gamma", type=float, required=True, help="The discount factor")
    parser.add_argument("-alpha", type=float, required=True, help="The learning rate")
    parser.add_argument("-epsilon", type=float, required=True, help="The initial epsilon-greedy probability")
    parser.add_argument("-decay", type=float, required=False, help="The epsilon-greedy probability decay")
    parser.add_argument("-episodes", type=int, required=True, help="The number of episodes")

    args = parser.parse_args()
    params = {
        "order": args.order,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
        "decay": args.decay,
        "episodes": args.episodes
    }

    statistics = {"errors": [], "stopping_points": [], "utilities": []}

    metareasoning_env = env.Environment(args.problem, ALPHA, BETA, INCREMENT)

    if not args.function:
        prakhar = table_agent.Agent(params, metareasoning_env)
    elif args.function == "fourier":
        prakhar = fourier_agent.Agent(params, metareasoning_env)
    else:
        print("Encountered an unrecognized function approximation:", args.function)
        sys.exit()

    if args.method == "q-learning":
        prakhar.run_q_learning(statistics)
    elif args.method == "sarsa":
        prakhar.run_sarsa(statistics)
    else:
        print("Encountered an unrecognized reinforcement learning method:", args.method)
        sys.exit()

    plt.figure(figsize=(7, 3))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["grid.linestyle"] = "-"
    plt.xlabel("Episodes")
    plt.ylabel("Error (%)")
    plt.grid(True)

    axis = plt.gca()
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)

    smoothed_errors = utils.get_smoothed_values(statistics["errors"], WINDOW_SIZE)
    plt.plot(range(len(smoothed_errors)), smoothed_errors)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
