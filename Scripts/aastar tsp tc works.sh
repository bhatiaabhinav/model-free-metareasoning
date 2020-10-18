# DQN Dynamic (W=2) N3-step DDQN
python -m MFMR.deepRL A2.0Astar-25tsp-v0 DQN 200000 --gamma=0.999 --double_dqn --algo_suffix=N3D_dynW2_tc10_s03 --tags s03 dyn W2 --framestack=20 --seed=0 --hiddens 128 64 --nsteps=3 --train_freq=1 --exp_buff_len=100000 --mb_size=64 --grad_clip=1 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=1000 --ep_anneal_steps=9900 --ep=0.01 --model_save_freq=10000 --no_render --no_monitor --overwrite --no_gpu

# Static (W=2) Baseline
python -m MFMR.deepRL A2.0Astar-25tsp-v0 Random 200000 --gamma=0.999 --double_dqn --algo_suffix=Static_W2_tc10_s03 --tags s03 static W2 --framestack=20 --seed=0 --hiddens 128 64 --nsteps=3 --train_freq=1 --exp_buff_len=100000 --mb_size=64 --grad_clip=1 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=1000 --ep_anneal_steps=9900 --ep=0.01 --model_save_freq=10000 --no_render --no_monitor --overwrite --no_gpu

# Random (W=2) Baseline
python -m MFMR.deepRL A2.0Astar-25tsp-v0 Random 200000 --gamma=0.999 --double_dqn --algo_suffix=Random_dynW2_tc10_s03 --tags s03 random W2 --framestack=20 --seed=0 --hiddens 128 64 --nsteps=3 --train_freq=1 --exp_buff_len=100000 --mb_size=64 --grad_clip=1 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=1000 --ep_anneal_steps=9900 --ep=0.01 --model_save_freq=10000 --no_render --no_monitor --overwrite --no_gpu

# Static (W=3) Baseline
python -m MFMR.deepRL A2.0Astar-25tsp-v0 Random 200000 --gamma=0.999 --double_dqn --algo_suffix=Static_W3_tc10_s03 --tags s03 static W3 --framestack=20 --seed=0 --hiddens 128 64 --nsteps=3 --train_freq=1 --exp_buff_len=100000 --mb_size=64 --grad_clip=1 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=1000 --ep_anneal_steps=9900 --ep=0.01 --model_save_freq=10000 --no_render --no_monitor --overwrite --no_gpu

# Random (W=3) Baseline
python -m MFMR.deepRL A2.0Astar-25tsp-v0 Random 200000 --gamma=0.999 --double_dqn --algo_suffix=Random_dynW3_tc10_s03 --tags s03 random W3 --framestack=20 --seed=0 --hiddens 128 64 --nsteps=3 --train_freq=1 --exp_buff_len=100000 --mb_size=64 --grad_clip=1 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=1000 --ep_anneal_steps=9900 --ep=0.01 --model_save_freq=10000 --no_render --no_monitor --overwrite --no_gpu





# Final Params
# ---------------
# (DQN fractions multiplied by 5 here)


# Playing for 200K steps (10 hr) and 50 steps per ep => 4000 episodes.

# Experience buffer should be 1/10th of total. (was 1/50th in DQN)
#   Exp_buff_len = 200K/10 steps = 20k steps. (1 hr) i.e. 400 episodes.

# Anneal over = exp buff len:
#   Anneal_over = 20k steps (1 hr) i.e. 400 episodes.

# 1/4th of exp buffer should be filled with random. (was 1/20th in DQN)
#   => min_explore for 20k/4 steps = 5k steps (15 mins) i.e. 100 episodes

# target net frequency 1/20th of experience buffer (was 1/100th in DQN)
#   => 20k/20 steps = 1k steps (3 mins) = 20 episodes

# increase learning rate to 0.00025
# keeping mb size 64 because a lot of noise in data

# epsilon? I want about 1 random 0.5 change. 1 random 0.1 change. i.e. 2 changes per ep. i.e. 2 in 50. ep = 2/50 = 0.04.
# lets keep it 0.05. => 2.5 changes per ep. Good :)

# save model 20 times in 200k steps. => save every 10k (30 mins) steps.

# experiment methodology:
#------------------------
# run 5 seeds (0-4) of our algo simulataneouly and get curves with train mean and std.
# test for 5k steps. over seeds 10-14.
# For baselines, run for 5k steps (15 mins) i.e. 100 episodes. for 5 seeds and get mean and std
# First run baselines.. evaluate over seeds 10-14


# DQN Dynamic (W=2) N3-step DDQN
python -m MFMR.deepRL A2.0Astar-25tsp-v0 DQN 200000 --gamma=0.999 --double_dqn --algo_suffix=seed0_N3D_dynW2_tc10_s03_final --tags s03 dyn W2 final --framestack=20 --seed=0 --hiddens 128 64 --nsteps=3 --train_freq=1 --exp_buff_len=200000 --mb_size=64 --lr=0.00025 --grad_clip=1 --target_q_freq=1000 --min_explore_steps=5000 --ep_anneal_steps=20000 --ep=0.05 --model_save_freq=10000 --no_render --no_monitor --overwrite --no_gpu
python -m MFMR.deepRL A2.0Astar-25tsp-v0 DQN 200000 --gamma=0.999 --double_dqn --algo_suffix=seed1_N3D_dynW2_tc10_s03_final --tags s03 dyn W2 final --framestack=20 --seed=1 --hiddens 128 64 --nsteps=3 --train_freq=1 --exp_buff_len=200000 --mb_size=64 --lr=0.00025 --grad_clip=1 --target_q_freq=1000 --min_explore_steps=5000 --ep_anneal_steps=20000 --ep=0.05 --model_save_freq=10000 --no_render --no_monitor --overwrite --no_gpu
python -m MFMR.deepRL A2.0Astar-25tsp-v0 DQN 200000 --gamma=0.999 --double_dqn --algo_suffix=seed2_N3D_dynW2_tc10_s03_final --tags s03 dyn W2 final --framestack=20 --seed=2 --hiddens 128 64 --nsteps=3 --train_freq=1 --exp_buff_len=200000 --mb_size=64 --lr=0.00025 --grad_clip=1 --target_q_freq=1000 --min_explore_steps=5000 --ep_anneal_steps=20000 --ep=0.05 --model_save_freq=10000 --no_render --no_monitor --overwrite --no_gpu
python -m MFMR.deepRL A2.0Astar-25tsp-v0 DQN 200000 --gamma=0.999 --double_dqn --algo_suffix=seed3_N3D_dynW2_tc10_s03_final --tags s03 dyn W2 final --framestack=20 --seed=3 --hiddens 128 64 --nsteps=3 --train_freq=1 --exp_buff_len=200000 --mb_size=64 --lr=0.00025 --grad_clip=1 --target_q_freq=1000 --min_explore_steps=5000 --ep_anneal_steps=20000 --ep=0.05 --model_save_freq=10000 --no_render --no_monitor --overwrite --no_gpu
python -m MFMR.deepRL A2.0Astar-25tsp-v0 DQN 200000 --gamma=0.999 --double_dqn --algo_suffix=seed4_N3D_dynW2_tc10_s03_final --tags s03 dyn W2 final --framestack=20 --seed=4 --hiddens 128 64 --nsteps=3 --train_freq=1 --exp_buff_len=200000 --mb_size=64 --lr=0.00025 --grad_clip=1 --target_q_freq=1000 --min_explore_steps=5000 --ep_anneal_steps=20000 --ep=0.05 --model_save_freq=10000 --no_render --no_monitor --overwrite --no_gpu

# Static (W=2) Baseline
python -m MFMR.deepRL A2.0Astar-25tsp-v0 Random 5000 --algo_suffix=Static_seed10_W2_tc10_s03_final --tags s03 static W2 baseline final --seed=10 --no_render --no_monitor --overwrite --no_gpu

# Random (W=2) Baseline
python -m MFMR.deepRL A2.0Astar-25tsp-v0 Random 5000 --algo_suffix=Random_seed10_dynW2_tc10_s03_final --tags s03 random W2 baseline final --seed=10 --no_render --no_monitor --overwrite --no_gpu

# Static (W=3) Baseline
python -m MFMR.deepRL A2.0Astar-25tsp-v0 Random 5000 --algo_suffix=Static_seed10_W3_tc10_s03_final --tags s03 static W3 baseline final --seed=10 --no_render --no_monitor --overwrite --no_gpu

# Random (W=3) Baseline
python -m MFMR.deepRL A2.0Astar-25tsp-v0 Random 5000 --algo_suffix=Random_seed10_dynW3_tc10_s03_final --tags s03 random W2 baseline final --seed=10 --no_render --no_monitor --overwrite --no_gpu
