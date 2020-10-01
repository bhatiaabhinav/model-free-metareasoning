# DQN Dynamic (W=2) N3-step DDQN
python -m MFMR.deepRL A2.0Astar-25tsp-v0 DQN 100000 --gamma=0.999 --double_dqn --algo_suffix=N3D_dynW2_tc10_s03 --tags s03 dyn W2 --framestack=20 --seed=0 --hiddens 128 64 --nsteps=3 --train_freq=1 --exp_buff_len=100000 --mb_size=64 --grad_clip=1 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=1000 --ep_anneal_steps=9900 --ep=0.01 --no_render --no_monitor --overwrite --no_gpu

# Static (W=2) Baseline
python -m MFMR.deepRL A2.0Astar-25tsp-v0 Random 100000 --gamma=0.999 --double_dqn --algo_suffix=Random_W2_tc10_s03 --tags s03 W2 --framestack=20 --seed=0 --hiddens 128 64 --nsteps=3 --train_freq=1 --exp_buff_len=100000 --mb_size=64 --grad_clip=1 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=1000 --ep_anneal_steps=9900 --ep=0.01 --no_render --no_monitor --overwrite --no_gpu

# Random (W=2) Baseline
python -m MFMR.deepRL A2.0Astar-25tsp-v0 Random 100000 --gamma=0.999 --double_dqn --algo_suffix=Random_dynW2_tc10_s03 --tags s03 dyn W2 --framestack=20 --seed=0 --hiddens 128 64 --nsteps=3 --train_freq=1 --exp_buff_len=100000 --mb_size=64 --grad_clip=1 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=1000 --ep_anneal_steps=9900 --ep=0.01 --no_render --no_monitor --overwrite --no_gpu

# Static (W=3) Baseline
python -m MFMR.deepRL A2.0Astar-25tsp-v0 Random 100000 --gamma=0.999 --double_dqn --algo_suffix=Random_W3_tc10_s03 --tags s03 W3 --framestack=20 --seed=0 --hiddens 128 64 --nsteps=3 --train_freq=1 --exp_buff_len=100000 --mb_size=64 --grad_clip=1 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=1000 --ep_anneal_steps=9900 --ep=0.01 --no_render --no_monitor --overwrite --no_gpu