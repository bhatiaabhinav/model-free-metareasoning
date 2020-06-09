from gym.envs import register

from MFMR.algos.file_algo import FileAlgo

register('FileAlgoTest-v0', entry_point='MFMR.monitors.async_algo_monitor:AsyncAlgoMonitor', kwargs={
    'alpha': 200,
    'beta': 0.3,
    'monitoring_interval': 1 / 60,  # 60 FPS
    'algo_cls': FileAlgo,
    'problem_file_path': "problems/test.json",
    'increment': 5,
    'discretization': True
})
