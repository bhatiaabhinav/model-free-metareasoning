w=1 python -m MFMR.deepRL AWAstar-citynav-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W1_tc3_sim_ijcai --tags static W1 tc3 sim ijcai c150 p150 nc3 pc3 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

w=1.5 python -m MFMR.deepRL AWAstar-citynav-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W1.5_tc3_sim_ijcai --tags static W1.5 tc3 sim ijcai c150 p150 nc3 pc3 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

w=2 python -m MFMR.deepRL AWAstar-citynav-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W2_tc3_sim_ijcai --tags static W2 tc3 sim ijcai c150 p150 nc3 pc3 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

w=3 python -m MFMR.deepRL AWAstar-citynav-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W3_tc3_sim_ijcai --tags static W3 tc3 sim ijcai c150 p150 nc3 pc3 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

w=4 python -m MFMR.deepRL AWAstar-citynav-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W4_tc3_sim_ijcai --tags static W4 tc3 sim ijcai c150 p150 nc3 pc3 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

w=5 python -m MFMR.deepRL AWAstar-citynav-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_W5_tc3_sim_ijcai --tags static W5 tc3 sim ijcai c150 p150 nc3 pc3 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

sleep 5

Adjust=True python -m MFMR.deepRL AWAstar-citynav-v0 Random 200000 --num_episodes_to_run=500 --algo_suffix=seed10_random_tc3_sim_ijcai --tags random tc3 sim ijcai c150 p150 nc3 pc3 --seed=10 --RPE_av_over=500 --no_render --no_monitor --overwrite &

