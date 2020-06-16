from gym.envs import register

from MFMR.algos.file_algo import FileAlgo
from MFMR.algos.tsp import Tsp

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
