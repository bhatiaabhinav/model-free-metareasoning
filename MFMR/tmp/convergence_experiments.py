import matplotlib.pyplot as plt

import env
import table_agent
import utils

PROBLEM_DIRECTORY = "problems/"
RESULTS_DIRECTORY = "statistics/"

PROBLEM_FILE = "test.json"

WINDOW_SIZE = 50
PLOT_WINDOW_SIZE = 350

ALPHA = 100
BETA = 0.08
INCREMENT = 1

PROBLEM_FILE_PATH = PROBLEM_DIRECTORY + PROBLEM_FILE


def run_tabular_sarsa_experiments(params):
    statistics = {"errors": [], "stopping_points": [], "utilities": []}

    filename = RESULTS_DIRECTORY + "tabular-sarsa-[{}]-[{}]-{}".format(params["alpha"], params["epsilon"], PROBLEM_FILE)

    metareasoning_env = env.Environment(PROBLEM_FILE_PATH, ALPHA, BETA, INCREMENT)
    prakhar = table_agent.Agent(params, metareasoning_env)
    prakhar.run_sarsa(statistics)

    utils.save(filename, statistics)

    return utils.get_results(statistics["errors"], WINDOW_SIZE, PLOT_WINDOW_SIZE)


def run_tabular_q_learning_experiments(params):
    statistics = {"errors": [], "stopping_points": [], "utilities": []}

    filename = RESULTS_DIRECTORY + "tabular-q-[{}]-[{}]-{}".format(params["alpha"], params["epsilon"], PROBLEM_FILE)

    metareasoning_env = env.Environment(PROBLEM_FILE_PATH, ALPHA, BETA, INCREMENT)
    prakhar = table_agent.Agent(params, metareasoning_env)
    prakhar.run_q_learning(statistics)

    utils.save(filename, statistics)

    return utils.get_results(statistics["errors"], WINDOW_SIZE, PLOT_WINDOW_SIZE)


def run():
    print("Convergence experiment [{}] with [alpha = {}, beta = {}, increment = {}]".format(PROBLEM_FILE, ALPHA, BETA, INCREMENT))

    tabular_sarsa_results = run_tabular_sarsa_experiments({
        "alpha": 0.1,
        "epsilon": 0.1,
        "gamma": 1.0,
        "decay": 0.999,
        "episodes": 5000
    })
    print("Error: {} +/- {}".format(tabular_sarsa_results["mean_error"], tabular_sarsa_results["standard_deviation_error"]))

    tabular_q_learning_results = run_tabular_q_learning_experiments({
        "alpha": 0.1,
        "epsilon": 0.1,
        "gamma": 1.0,
        "decay": 0.999,
        "episodes": 5000
    })
    print("Error: {} +/- {}".format(tabular_q_learning_results["mean_error"], tabular_q_learning_results["standard_deviation_error"]))

    plt.figure(figsize=(7, 3))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["grid.linestyle"] = "-"
    plt.xlabel("Episodes")
    plt.ylabel("Error")
    plt.grid(True)

    axis = plt.gca()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    p1 = plt.plot(range(len(tabular_sarsa_results["smoothed_errors"])), tabular_sarsa_results["smoothed_errors"], color="r")
    p2 = plt.plot(range(len(tabular_q_learning_results["smoothed_errors"])), tabular_q_learning_results["smoothed_errors"], color="b")

    plt.legend((p1[0], p2[0]), ("SARSA(Table)", "Q-learning(Table)"), loc=1, ncol=2)

    plt.tight_layout()
    plt.show()


def test():
    metareasoning_env = env.Environment(PROBLEM_FILE_PATH, ALPHA, BETA, INCREMENT)

    quality, time = metareasoning_env.reset()
    qualities = [quality]

    utility = utils.get_time_dependent_utility(quality, time, ALPHA, BETA)
    utilities = [utility]

    while True:
        (quality, time), _, is_episode_done = metareasoning_env.step(metareasoning_env.CONTINUE_ACTION)

        qualities.append(quality)
        utilities.append(utils.get_time_dependent_utility(quality, time, ALPHA, BETA))

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


def main():
    run()


if __name__ == "__main__":
    main()
