# model-free-metareasoning


## Installation Instruction

```bash
git clone https://github.com/justinsvegliato/model-free-metareasoning.git
cd model-free-metareasoning
python3 -m venv env  # create virtual environment
source env/bin/activate  # activate virtual environment
pip install wheel
pip install -r requirements.txt
pip install -e .
```

# Test Installation

After activating virtual environment, from repo folder:
```bash
pytest Tests/
```


## Running with Deep RL as the metareasoner

Install [RL-v2](https://github.com/bhatiaabhinav/RL-v2) library :
```bash
cd ./..  # get out of repo folder
git clone git+https://github.com/bhatiaabhinav/RL-v2.git
pip install -e RL-v2
cd model-free-metareasoning
```
You will optionally need a [Weights & Biases](https://www.wandb.com/) account to visualize RL runs i.e. to see graphs, comparisons etc. You will be asked to login with your account token on running an experiment for the first time on your machine.

Example run commands:

[DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) (for discrete action space):
```bash
python -m MFMR.deepRL RRTStar-Easy-v0 DQN 100000 --algo_suffix=test_experiment --seed=0 --hiddens 64 32 --train_freq=1 --target_q_freq=2000 --min_explore_steps=10000 --ep_anneal_steps=10000 --ep=0.01 --no_render
```
DQN = Q-Learning + Neural Net to approximate Q + experience replay for data efficiency + ep-greedy exploration.

[SAC](https://arxiv.org/abs/1812.05905) (for continous action space):
```bash
python -m MFMR.deepRL RRTStar-Easy-v0 SAC 100000 --algo_suffix=text_experiment --seed=0 --hiddens 64 32 --train_freq=1--min_explore_steps=10000 --grad_clip=1 --no_render
```
SAC = [DDPG](https://arxiv.org/abs/1509.02971) + stochastic actor + entropy bonus for exploration. DDPG = DQN for continuous action spaces where argmax_a Q(s,a) is learnt via gradient ascent.


Easy introduction to DDPG & SAC is at [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ddpg.html).


To get [list of hyperparams](https://github.com/bhatiaabhinav/RL-v2/blob/master/RL/algorithms/__init__.py):
```bash
python -m MFMR.deepRL -h
```
