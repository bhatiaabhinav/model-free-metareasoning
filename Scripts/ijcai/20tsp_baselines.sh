w=1 python -m MFMR.deepRL A2.0Astar-20tsp-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W1_tc5_sim_20tsp_ijcai --tags static W1 tc5 sim ijcai tsp15to25 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

w=1.5 python -m MFMR.deepRL A2.0Astar-20tsp-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W1.5_tc5_sim_20tsp_ijcai --tags static W1.5 tc5 sim ijcai tsp15to25 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

w=2 python -m MFMR.deepRL A2.0Astar-20tsp-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W2_tc5_sim_20tsp_ijcai --tags static W2 tc5 sim ijcai tsp15to25 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

w=3 python -m MFMR.deepRL A2.0Astar-20tsp-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W3_tc5_sim_20tsp_ijcai --tags static W3 tc5 sim ijcai tsp15to25 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

w=4 python -m MFMR.deepRL A2.0Astar-20tsp-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W4_tc5_sim_20tsp_ijcai --tags static W4 tc5 sim ijcai tsp15to25 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

w=5 python -m MFMR.deepRL A2.0Astar-20tsp-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W5_tc5_sim_20tsp_ijcai --tags static W5 tc5 sim ijcai tsp15to25 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

Adjust=True python -m MFMR.deepRL A2.0Astar-20tsp-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_random_tc5_sim_20tsp_ijcai --tags random tc5 sim ijcai tsp15to25 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

