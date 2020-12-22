
import numpy as np
from gym.envs import register
from MFMR.algos.aastar import AAstar
from MFMR.algos.city_nav import CityNavigation
from MFMR.algos.file_algo import FileAlgo
from MFMR.algos.n_puzzle import NPuzzle
from MFMR.algos.tsp import Tsp
from MFMR.algos.tsp_search_prob import TSPProblem

register('FileAlgo-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor', kwargs={
    'alpha': 200,
    'beta': 0.1,
    'monitoring_interval': 1 / 60,  # 60 FPS
    'algo_cls': FileAlgo,
    'problem_file_path': "problems/test.json",
    'increment': 3,
    'discretization': True
})

register('FileAlgoBeta0-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor', kwargs={
    'alpha': 200,
    'beta': 0.1,
    'monitoring_interval': 1 / 60,  # 60 FPS
    'algo_cls': FileAlgo,
    'problem_file_path': "problems/test.json",
    'increment': 3,
    'discretization': True
})


register('FileAlgo-Cont-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor', kwargs={
    'alpha': 200,
    'beta': 0.1,
    'monitoring_interval': 1 / 60,  # 60 FPS
    'algo_cls': FileAlgo,
    'problem_file_path': "problems/test.json",
    'increment': 3,
    'discretization': False
})

register('FileAlgoBeta0-Cont-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor', kwargs={
    'alpha': 200,
    'beta': 0,
    'monitoring_interval': 1 / 60,  # 60 FPS
    'algo_cls': FileAlgo,
    'problem_file_path': "problems/test.json",
    'increment': 3,
    'discretization': False
})

register('Tsp30-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor', kwargs={
    'alpha': 200,
    'beta': 0.5,
    'monitoring_interval': 1 / 10,
    'algo_cls': Tsp,
    'instances_directory': 'problems/30-tsp',
    'index_file_path': 'problems/30-tsp/optimal-costs.csv',
    'discretization': True
})

register('Tsp30-Cont-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor', kwargs={
    'alpha': 200,
    'beta': 0.5,
    'monitoring_interval': 1 / 10,
    'algo_cls': Tsp,
    'instances_directory': 'problems/30-tsp',
    'index_file_path': 'problems/30-tsp/optimal-costs.csv',
    'discretization': False
})

register('Tsp30Beta0-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor', kwargs={
    'alpha': 200,
    'beta': 0,
    'monitoring_interval': 1 / 10,
    'algo_cls': Tsp,
    'instances_directory': 'problems/30-tsp',
    'index_file_path': 'problems/30-tsp/optimal-costs.csv',
    'discretization': True
})

register('Tsp30Beta0-Cont-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor', kwargs={
    'alpha': 200,
    'beta': 0,
    'monitoring_interval': 1 / 10,
    'algo_cls': Tsp,
    'instances_directory': 'problems/30-tsp',
    'index_file_path': 'problems/30-tsp/optimal-costs.csv',
    'discretization': False
})


# for w in np.arange(1, 3.25, 0.1):
#     w = np.round(w, 2)
#     for n in range(1, 9):
#         register(f'A{w}Astar-{n}puzzle-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor',
#                  kwargs={
#                      'alpha': 1000,
#                      # 'beta_options': [beta - 0.1, beta, beta + 0.1],
#                      'beta_options': [0.0],
#                      'stop_action_available': False,
#                      'monitoring_interval': 1 / 5,
#                      'observe_beta': False,
#                      'algo_cls': AAstar,
#                      'weight': w,
#                      'weight_max': 5,
#                      'weight_interval': 0.25,
#                      'time_max': 10,
#                      'ref_nodes_budget': 6000,
#                      'simulate_ref_machine': True,
#                      'adjust_weight': True,
#                      'observe_ub': True,
#                      'search_problem_cls': NPuzzle,
#                      'N_range': [n, n + 1],
#                      'min_manhattan': 45,
#                      'max_manhattan': 50,
#                      'inverse': False
#                  }
#                  )

# for w in np.arange(1, 3.25, 0.1):
#     w = np.round(w, 2)
#     for n in range(1, 9):
#         register(f'A{w}Astar-inv-{n}puzzle-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor',
#                  kwargs={
#                      'alpha': 1000,
#                      # 'beta_options': [beta - 0.1, beta, beta + 0.1],
#                      'beta_options': [0.0],
#                      'stop_action_available': False,
#                      'monitoring_interval': 1 / 5,
#                      'observe_beta': False,
#                      'algo_cls': AAstar,
#                      'weight': w,
#                      'weight_max': 5,
#                      'weight_interval': 0.25,
#                      'time_max': 10,
#                      'ref_nodes_budget': 6000,
#                      'simulate_ref_machine': True,
#                      'adjust_weight': True,
#                      'observe_ub': True,
#                      'search_problem_cls': NPuzzle,
#                      'N_range': [n, n + 1],
#                      'min_manhattan': 50,
#                      'max_manhattan': 55,
#                      'inverse': True
#                  }
#                  )


# for w in np.arange(1, 3.25, 0.1):
#     w = np.round(w, 2)
#     for n in range(10, 45, 5):
#         register(f'A{w}Astar-{n}tsp-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor',
#                  kwargs={
#                      'alpha': 1000,
#                      # 10s, 9s, 7s, 6s, 4s
#                      #  'beta_options': [0.3, 0.4, 0.5, 0.6, 0.7],
#                      # "beta_options": [0],
#                      'beta_options': [0.0],
#                      'stop_action_available': False,
#                      'monitoring_interval': 1 / 5,
#                      'observe_beta': False,
#                      'algo_cls': AAstar,
#                      'weight': w,
#                      'weight_max': 5,
#                      'weight_interval': 0.25,
#                      'time_max': 10,
#                      'ref_nodes_budget': 6000,
#                      'simulate_ref_machine': True,
#                      'adjust_weight': True,
#                      'observe_ub': True,
#                      'search_problem_cls': TSPProblem,
#                      'N_range': [n - 5, n + 5],
#                      #  'N_options': [10],
#                      'sparsity_range': [0.0, 0.3]
#                  }
#                  )


# for w in np.arange(1, 3.25, 0.1):
#     w = np.round(w, 2)
#     for n in range(10, 21):  # TODO: city navigation size range
#         register(f'A{w}Astar-{n}citynav-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor',
#                  kwargs={
#                      'alpha': 1000,
#                      # 10s, 9s, 7s, 6s, 4s
#                      #  'beta_options': [0.3, 0.4, 0.5, 0.6, 0.7],
#                      # "beta_options": [0],
#                      'beta_options': [0.0],  # time cost
#                      'stop_action_available': False,  # False means time contract setting
#                      'monitoring_interval': 1 / 5,
#                      'observe_beta': False,  # set to true to observe timecost
#                      'algo_cls': AAstar,
#                      'weight': w,
#                      'weight_max': 5,
#                      'weight_interval': 0.25,
#                      'time_max': 10,
#                      'ref_nodes_budget': 6000,
#                      'simulate_ref_machine': True,
#                      'adjust_weight': True,
#                      'observe_ub': True,
#                      'search_problem_cls': CityNavigation,
#                      'example_arg1': n,
#                      'example_kwarg1': 20
#                  }
#                  )


for w in np.arange(1, 3.25, 0.1):
    w = np.round(w, 2)
    for n in range(1, 9):
        register(f'A{w}Astar-{n}puzzle-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor',
                 kwargs={
                     'alpha': 1000,
                     # 'beta_options': [beta - 0.1, beta, beta + 0.1],
                     'beta_options': [0.0],
                     'stop_action_available': False,
                     'monitoring_interval': 1 / 5,
                     'observe_beta': False,
                     'algo_cls': AAstar,
                     'weight': 5,
                     'weight_max': 5,
                     'weight_interval': 0.25,
                     'time_max': 10,
                     'ref_nodes_budget': 6000,
                     'simulate_ref_machine': True,
                     'adjust_weight': False,
                     'random_adjust_weight': False,
                     'observe_ub': True,
                     'search_problem_cls': NPuzzle,
                     'N_range': [n, n],
                     'min_manhattan': 35,
                     'max_manhattan': 45,
                     'inverse': False
                 }
                 )

for w in np.arange(1, 3.25, 0.1):
    w = np.round(w, 2)
    for n in range(1, 9):
        register(f'A{w}Astar-inv-{n}puzzle-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor',
                 kwargs={
                     'alpha': 1000,
                     # 'beta_options': [beta - 0.1, beta, beta + 0.1],
                     'beta_options': [0.0],
                     'stop_action_available': False,
                     'monitoring_interval': 1 / 5,
                     'observe_beta': False,
                     'algo_cls': AAstar,
                     'weight': w,
                     'weight_max': 5,
                     'weight_interval': 0.25,
                     'time_max': 10,
                     'ref_nodes_budget': 6000,
                     'simulate_ref_machine': True,
                     'adjust_weight': True,
                     'random_adjust_weight': True,
                     'observe_ub': True,
                     'search_problem_cls': NPuzzle,
                     'N_range': [n, n],
                     'min_manhattan': 35,
                     'max_manhattan': 45,
                     'inverse': True
                 }
                 )


for w in np.arange(1, 3.25, 0.1):
    w = np.round(w, 2)
    for n in range(10, 45, 5):
        register(f'A{w}Astar-{n}tsp-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor',
                 kwargs={
                     'alpha': 1000,
                     # 10s, 9s, 7s, 6s, 4s
                     #  'beta_options': [0.3, 0.4, 0.5, 0.6, 0.7],
                     # "beta_options": [0],
                     'beta_options': [0.0],
                     'stop_action_available': False,
                     'monitoring_interval': 1 / 5,
                     'observe_beta': False,
                     'algo_cls': AAstar,
                     'weight': w,
                     'weight_max': 5,
                     'weight_interval': 0.25,
                     'time_max': 10,
                     'ref_nodes_budget': 6000,
                     'simulate_ref_machine': True,
                     'adjust_weight': True,
                     'random_adjust_weight': True,
                     'observe_ub': True,
                     'search_problem_cls': TSPProblem,
                     'N_range': [n - 5, n + 5],
                     #  'N_options': [10],
                     'sparsity_range': [0.0, 0.3]
                 }
                 )


register(f'AWAstar-citynav-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor',
         kwargs={
             'alpha': 1000,
             # 10s, 9s, 7s, 6s, 4s
             #  'beta_options': [0.3, 0.4, 0.5, 0.6, 0.7],
             # "beta_options": [0],
             'beta_options': [0.0],  # time cost
             'stop_action_available': False,  # False means time contract setting
             'monitoring_interval': 1 / 5,
             'observe_beta': False,  # set to true to observe timecost
             'algo_cls': AAstar,
             'weight': 1,
             'weight_max': 5,
             'weight_interval': 0.25,
             'time_max': 4,
             'ref_nodes_budget': 2400,
             'simulate_ref_machine': True,
             'adjust_weight': True,
             'random_adjust_weight': True,
             'observe_ub': False,
             'search_problem_cls': CityNavigation,
             'n_c': 3,
             'n_p': 3,
             'num_cities': 150,
             'num_places': 150
         }
         )
