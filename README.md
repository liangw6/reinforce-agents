# A3C Agent in OpenAI Gym
This repository implements the popular Asynchronous advantage actor-critic algorithm [1] in PyTorch. It also experiments applying the agent to two different OpenAI Gym tasks, CartPole and LunarLander.

## Code structure
* `agents.py` implements REINFORCE agents, A3C agent and a random agent 
* `main.py` implements training and saves the model afterwards
* `load_and_render.py` can load the model and run eval() for rendering. It also prints average reward

The code also uses the framework [2]

## Reference
[1] Mnih, Volodymyr, et al. “Asynchronous Methods for Deep Reinforcement Learning.” ArXiv:1602.01783 [Cs], June 2016. arXiv.org, http://arxiv.org/abs/1602.01783.
[2] Redmon, Joseph. Pjreddie/Rl-Hw. 2018. 2020. GitHub, https://github.com/pjreddie/rl-hw.