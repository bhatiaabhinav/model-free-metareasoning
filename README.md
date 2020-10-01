# model-free-metareasoning


## Installation Instruction

```bash
git clone https://github.com/justinsvegliato/model-free-metareasoning.git
cd model-free-metareasoning
python3 -m venv env  # create virtual environment
source env/bin/activate  # activate virtual environment
pip install wheel
pip install -r requirements.txt
```

## Install Deep RL as the metareasoner

Install [RL-v2](https://github.com/bhatiaabhinav/RL-v2) library :
```bash
cd ./..  # get out of repo folder
git clone git+https://github.com/bhatiaabhinav/RL-v2.git
pip install -e RL-v2
cd model-free-metareasoning
pip install -e .
```

You will need a [Weights & Biases](https://www.wandb.com/) account to visualize RL runs i.e. to see graphs, comparisons etc. You will be asked to login with your account token on running an experiment for the first time on your machine.


# Test Installation

After activating virtual environment, from repo folder:
```bash
pytest Tests/
```

# Example run commands:

```bash
python -m MFMR.deepRL A2.0Astar-25tsp-v0 DQN 100000 --gamma=0.999 --double_dqn --algo_suffix=N3D_dynW2_tc10_s03 --tags s03 dyn W2 --framestack=20 --seed=0 --hiddens 128 64 --nsteps=3 --train_freq=1 --exp_buff_len=100000 --mb_size=64 --grad_clip=1 --target_q_freq=1 --target_q_polyak=0.999 --min_explore_steps=1000 --ep_anneal_steps=9900 --ep=0.01 --no_render --no_monitor --overwrite --no_gpu
```

To get list of hyperparams, see RL-v2 [readme.md](https://github.com/bhatiaabhinav/RL-v2/blob/master/readme.md)