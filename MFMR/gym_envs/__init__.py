from gym.envs import register

register('RRT*-Easy', entry_point='MMFR.gym_envs.rrt_star:RRTStar', kwargs={
    'gamma': 0.99,
    'map_size': (1, 1),
    'random_map_every_reset': False,
    'n_obstacles': 2
})

register('RRT*-Medium', entry_point='MMFR.gym_envs.rrt_star:RRTStar', kwargs={
    'gamma': 0.99,
    'map_size': (1, 1),
    'random_map_every_reset': False,
    'n_obstacles': 1
})

register('RRT*-Hard', entry_point='MMFR.gym_envs.rrt_star:RRTStar', kwargs={
    'gamma': 0.99,
    'map_size': (1, 1),
    'random_map_every_reset': False,
    'n_obstacles': 2
})

register('RRT*-Easy-Random', entry_point='MMFR.gym_envs.rrt_star:RRTStar', kwargs={
    'gamma': 0.99,
    'map_size': (1, 1),
    'random_map_every_reset': True,
    'n_obstacles': 2
})

register('RRT*-Medium-Random', entry_point='MMFR.gym_envs.rrt_star:RRTStar', kwargs={
    'gamma': 0.99,
    'map_size': (1, 1),
    'random_map_every_reset': True,
    'n_obstacles': 1
})

register('RRT*-Hard-Random', entry_point='MMFR.gym_envs.rrt_star:RRTStar', kwargs={
    'gamma': 0.99,
    'map_size': (1, 1),
    'random_map_every_reset': True,
    'n_obstacles': 2
})
