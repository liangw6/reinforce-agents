
"""

Main entrance for running rl agents in environments

Framework for individual classes are adapted
from https://github.com/pjreddie/rl-hw/blob/master/rl_hw_problems.ipynb

"""

from agents import *

def run_iteration(mode, N, agent, gen, horizon=None, render=False):
    train = mode == 'train'
    if train:
        agent.train()
    else:
        agent.eval()

    states, actions, rewards = zip(*[gen(horizon=horizon, render=render) for _ in range(N)])

    loss = None
    if train:
        loss = agent.learn(states, actions, rewards)

    reward = sum([r.sum() for r in rewards]) / N

    return reward, loss

lr = 1e-3
weight_decay = 1e-4

env_name = 'CartPole-v1'
#env_name = 'LunarLander-v2'
e = Pytorch_Gym_Env(env_name)
state_dim = e.observation_space.shape[0]
action_dim = e.action_space.n

# Choose what agent to use
#agent = REINFORCE(state_dim, action_dim, lr=lr, weight_decay=weight_decay)
agent = A3C(state_dim, action_dim, lr=lr, weight_decay=weight_decay)

total_episodes = 0
print(agent) # Let's take a look at what we're working with...


# Create a
gen = Generator(e, agent)

num_iter = 100
num_train = 10
num_eval = 10  # dont change this
for itr in range(num_iter):
    # agent.model.epsilon = epsilon * epsilon_decay ** (total_episodes / epsilon_decay_episodes)
    # print('** Iteration {}/{} **'.format(itr+1, num_iter))
    train_reward, train_loss = run_iteration('train', num_train, agent, gen)
    eval_reward, _ = run_iteration('eval', num_eval, agent, gen)
    total_episodes += num_train
    print(
        'Ep:{}: reward={:.3f}, loss={:.3f}, eval={:.3f}'.format(total_episodes, train_reward, train_loss, eval_reward))

    if eval_reward > 499 and env_name == 'CartPole-v1':  # dont change this
        print('Success!!! You have solved cartpole task! Time for a bigger challenge!')

    # save model
print('Done')

# You can visualize your policy at any time
run_iteration('eval', 1, agent, gen, render=True)