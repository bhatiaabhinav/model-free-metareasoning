Adjust=True
GraphFreq=10

# 15puzzle
# --------
Adjust=True python -m MFMR.deepRL A2.0Astar-4puzzle-v0 DQN 500000 --num_episodes_to_run=500 --RPE_av_over=500 --gamma=0.99 --double_dqn --dqn_mse_loss --algo_suffix=seed10_eval_seed0_dyn_tc10_sim_4puzzle_ijcai --tags evaluate dyn sim ijcai 4puzzle --framestack=5 --seed=10 --hiddens 64 32 --nsteps=1 --train_freq=1 --exp_buff_len=500000 --mb_size=64 --grad_clip=10 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=10000 --ep_anneal_steps=40000 --ep=0.0 --model_save_freq=10000 --no_render --no_monitor --no_gpu --eval_mode --eval_run=DQN_seed0_dyn_tc10_sim_4puzzle_ijcai &

sleep 5

# 15-inv-puzzle
# --------------
Adjust=True python -m MFMR.deepRL A2.0Astar-inv-4puzzle-v0 DQN 500000 --num_episodes_to_run=500 --RPE_av_over=500 --gamma=0.99 --double_dqn --dqn_mse_loss --algo_suffix=seed10_eval_seed0_dyn_tc10_sim_4puzzle_ijcai --tags evaluate dyn sim ijcai 4puzzle --framestack=5 --seed=10 --hiddens 64 32 --nsteps=1 --train_freq=1 --exp_buff_len=500000 --mb_size=64 --grad_clip=10 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=10000 --ep_anneal_steps=40000 --ep=0.0 --model_save_freq=10000 --no_render --no_monitor --no_gpu --eval_mode --eval_run=DQN_seed0_dyn_tc10_sim_4puzzle_ijcai &

sleep 5

# 20tsp
# -----
Adjust=True python -m MFMR.deepRL A2.0Astar-20tsp-v0 DQN 500000 --num_episodes_to_run=500 --RPE_av_over=500 --gamma=0.99 --double_dqn --dqn_mse_loss --algo_suffix=seed10_eval_seed0_dyn_tc5_sim_20tsp_ijcai --tags evaluate dyn tc5 sim ijcai tsp15to25 --framestack=5 --seed=0 --hiddens 64 32 --nsteps=1 --train_freq=1 --exp_buff_len=250000 --mb_size=64 --grad_clip=10 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=5000 --ep_anneal_steps=20000 --ep=0.0 --model_save_freq=10000 --no_render --no_monitor --no_gpu --eval_mode --eval_run=DQN_seed0_dyn_tc5_sim_20tsp_ijcai &

sleep 5

# 150cnp
# ------
Adjust=True python -m MFMR.deepRL AWAstar-citynav-v0 DQN 200000 --num_episodes_to_run=500 --RPE_av_over=500 --gamma=0.99 --double_dqn --dqn_mse_loss --algo_suffix=seed10_eval_seed0_dyn_tc3_sim_ijcai --tags evaluate dyn tc3 sim ijcai c150 p150 nc3 pc3 --framestack=5 --seed=0 --hiddens 64 32 --nsteps=1 --train_freq=1 --exp_buff_len=150000 --mb_size=64 --grad_clip=10 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=3000 --ep_anneal_steps=12000 --ep=0.0 --model_save_freq=10000 --no_render --no_monitor --no_gpu --eval_mode --eval_run=DQN_seed0_dyn_tc3_sim_ijcai &
