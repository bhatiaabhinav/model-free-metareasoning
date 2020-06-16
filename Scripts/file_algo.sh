# DQN on FileAlgo-Cont-v0
python -m deepRL FileAlgo-Cont-v0 DQN 50000 --algo_suffix=test_50k_steps --seed=0 --hiddens 32 --train_freq=4 --target_q_freq=2000 --min_explore_steps=500 --ep_anneal_steps=2500 --ep=0.01 --no_render
