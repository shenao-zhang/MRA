# Learning Meta Representations for Agents in Multi-Agent Reinforcement Learning
Code for paper "Learning Meta Representations for Agents in Multi-Agent Reinforcement Learning". 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Training and Evaluation

To obtain the meta representation in 3 Markov Games in the resource occupying environment, run:
 
```
python main.py --use_gpu --env_id resource --model_name multi_3_scenarios
```
 To train the MRA agents with different game settings, try to change the `task_list` and `spec_list` in `main.py`. 
 
 For example, change `6,9,12` to `3,6,12,24`.
 
## Acknowledgement
Some of our code is built upon the following repositories:

[MAAC](https://github.com/shariqiqbal2810/MAAC)

[EPC](https://github.com/qian18long/epciclr2020)