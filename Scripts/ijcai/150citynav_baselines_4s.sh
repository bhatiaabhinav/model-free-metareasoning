w=1 python -m MFMR.deepRL AWAstar-citynav-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W1_tc4_sim_ijcai --tags static W1 sim ijcai c150 p150 nc3 pc3 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

w=1.5 python -m MFMR.deepRL AWAstar-citynav-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W1.5_tc4_sim_ijcai --tags static W1.5 sim ijcai c150 p150 nc3 pc3 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

w=2 python -m MFMR.deepRL AWAstar-citynav-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W2_tc4_sim_ijcai --tags static W2 sim ijcai c150 p150 nc3 pc3 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

w=3 python -m MFMR.deepRL AWAstar-citynav-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W3_tc4_sim_ijcai --tags static W3 sim ijcai c150 p150 nc3 pc3 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

w=4 python -m MFMR.deepRL AWAstar-citynav-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W4_tc4_sim_ijcai --tags static W4 sim ijcai c150 p150 nc3 pc3 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

w=5 python -m MFMR.deepRL AWAstar-citynav-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W5_tc4_sim_ijcai --tags static W5 sim ijcai c150 p150 nc3 pc3 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

Adjust=True python -m MFMR.deepRL AWAstar-citynav-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_random_tc4_sim_ijcai --tags random sim ijcai c150 p150 nc3 pc3 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

