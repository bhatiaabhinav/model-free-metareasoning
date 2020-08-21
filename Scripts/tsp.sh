python -m MFMR.deepRL A2.0Astar-20tsp-B0.3-v0 DQN 100000 --double_dqn --algo_suffix=D_W2_ep0.01 --tags W2 --framestack=20 --seed=0 --hiddens 128 64 --train_freq=1 --exp_buff_len=100000 --mb_size=32 --grad_clip=1 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=1000 --ep_anneal_steps=19000 --ep=0.01 --no_render --overwrite --no_gpu


python -m MFMR.deepRL A2.0Astar-20tsp-B0.3-v0 DQN 100000 --double_dqn --algo_suffix=D_W2_ep0.01_evaluate --tags W2 evaluate --framestack=20 --seed=0 --hiddens 128 64 --train_freq=1 --exp_buff_len=100000 --mb_size=32 --grad_clip=1 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=1000 --ep_anneal_steps=19000 --ep=0.0 --no_render --overwrite --no_gpu --eval_mode --eval_run=DQN_D_W2_ep0.01 --monitor_video_freq=1


python -m MFMR.deepRL A2.0Astar-20tsp-B0.3-v0 DQN 100000 --double_dqn --algo_suffix=D_dynW_ep0.01 --tags dynW --framestack=20 --seed=0 --hiddens 128 64 --train_freq=1 --exp_buff_len=100000 --mb_size=32 --grad_clip=1 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=1000 --ep_anneal_steps=19000 --ep=0.01 --no_render --overwrite --no_gpu

python -m MFMR.deepRL A2.0Astar-20tsp-B0.3-v0 DQN 100000 --double_dqn --algo_suffix=D_dynW_ep0.01_evaluate --tags dynW evaluate --framestack=20 --seed=0 --hiddens 128 64 --train_freq=1 --exp_buff_len=100000 --mb_size=32 --grad_clip=1 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=1000 --ep_anneal_steps=19000 --ep=0.0 --no_render --overwrite --no_gpu --eval_mode --eval_run=DQN_D_dynW_ep0.01 --monitor_video_freq=1