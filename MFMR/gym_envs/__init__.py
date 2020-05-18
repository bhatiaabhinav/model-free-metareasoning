from gym.envs import register

register('RRTStar-Easy-v0', entry_point='MMFR.gym_envs.rrt_star:RRTStar', kwargs={
    'gamma': 0.99,
    'map_size': (1, 1),
    'random_map_every_reset': False,
    'n_obstacles': 0
})

register('RRTSrar-Medium-v0', entry_point='MMFR.gym_envs.rrt_star:RRTStar', kwargs={
    'gamma': 0.99,
    'map_size': (1, 1),
    'random_map_every_reset': False,
    'n_obstacles': 1
})

register('RRTStar-Hard-v0', entry_point='MMFR.gym_envs.rrt_star:RRTStar', kwargs={
    'gamma': 0.99,
    'map_size': (1, 1),
    'random_map_every_reset': False,
    'n_obstacles': 2
})

register('RRTStar-Easy-Random-v0', entry_point='MMFR.gym_envs.rrt_star:RRTStar', kwargs={
    'gamma': 0.99,
    'map_size': (1, 1),
    'random_map_every_reset': True,
    'n_obstacles': 0
})

register('RRTStar-Medium-Random-v0', entry_point='MMFR.gym_envs.rrt_star:RRTStar', kwargs={
    'gamma': 0.99,
    'map_size': (1, 1),
    'random_map_every_reset': True,
    'n_obstacles': 1
})

register('RRTStar-Hard-Random-v0', entry_point='MMFR.gym_envs.rrt_star:RRTStar', kwargs={
    'gamma': 0.99,
    'map_size': (1, 1),
    'random_map_every_reset': True,
    'n_obstacles': 2
})
