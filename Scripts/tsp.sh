# DQN on TSP30-Cont-v0
python -m deepRL Tsp30-Cont-v0 DQN 100000 --algo_suffix=test_100k_steps --seed=0 --hiddens 32 --train_freq=4 --target_q_freq=2000 --min_explore_steps=1000 --ep_anneal_steps=5000 --ep=0.01 --no_render
