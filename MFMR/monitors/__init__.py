import numpy as np
from gym.envs import register

from MFMR.algos.aastar import AAstar
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


for w in np.arange(1, 6.1, 0.5):
    w = np.round(w, 1)
    for n in range(1, 9):
        for difficulty_string in ['easy', 'medium', 'hard']:
            for beta in np.arange(0, 1.1, 0.1):
                beta = np.round(beta, 1)
                register(f'A{w}Astar-{n}puzzle-{difficulty_string}-B{beta}-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor',
                         kwargs={
                             'alpha': 200,
                             'beta_options': [beta - 0.1, beta, beta + 0.1],
                             'monitoring_interval': 1 / 10,
                             'algo_cls': AAstar,
                             'weight': w,
                             'weight_max': 6,
                             'weight_interval': 0.25,
                             'time_max': 15,
                             'adjust_weight': True,
                             'observe_ub': True,
                             'search_problem_cls': NPuzzle,
                             'n': n,
                             'difficulty_string': difficulty_string
                         }
                         )


for w in np.arange(1, 4.25, 0.25):
    w = np.round(w, 2)
    for n in range(10, 30, 5):
        register(f'A{w}Astar-{n}tsp-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor',
                 kwargs={
                     'alpha': 1000,
                     # 10s, 9s, 7s, 6s, 4s
                     'beta_options': [0.3, 0.4, 0.5, 0.6, 0.7],
                     #  'beta_options': [0],
                     'monitoring_interval': 1 / 4,
                     'observe_beta': True,
                     'algo_cls': AAstar,
                     'weight': w,
                     'weight_max': 4,
                     'weight_interval': 0.25,
                     'time_max': 10,
                     'adjust_weight': True,
                     'observe_ub': True,
                     'search_problem_cls': TSPProblem,
                     'N_options': [max(2, n-5), n, n + 5],
                     #  'N_options': [10],
                     'sparsity_range': [0, 0.75]
                 }
                 )
