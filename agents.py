"""

Core code for building rl agents

Framework for individual classes are adapted
from https://github.com/pjreddie/rl-hw/blob/master/rl_hw_problems.ipynb

But core code is mine

"""


from rlhw_util import *  # <-- look whats inside here - it could save you a lot of work!
import torch.nn.functional as F
from torch.optim import Adam
import math
import random
from collections import namedtuple

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()

        # NOTE: Fill in the code to define you policy
        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):

        # NOTE: Fill in the code to run a forward pass of your policy to get a distribution over actions (HINT: probabilities sum to 1)
        h1 = F.tanh(self.fc_1.forward(state))
        h2 = F.tanh(self.fc_2.forward(h1))
        out = F.softmax(self.fc_3.forward(h2))

        return out

    def get_policy(self, state):
        return Categorical(self(state))

    def get_action(self, state, greedy=None):
        # disable greedy. It's bad
        # if greedy is None:
        #     greedy = not self.training
        #
        # policy = self.get_policy(state)
        # return MLE(policy) if greedy else policy.sample()
        #
        policy = self.get_policy(state)
        return policy.sample()


class REINFORCE(nn.Module):

    def __init__(self, state_dim, action_dim, discount=0.97, lr=1e-3, weight_decay=1e-4, device='cuda'):
        super(REINFORCE, self).__init__()
        self.actor = Actor(state_dim, action_dim)

        self.baseline = nn.Linear(state_dim, 1)

        # NOTE: create an optimizer for the parameters of your actor (HINT: use the passed in lr and weight_decay args)
        self.optimizer = Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)

        self.discount = discount

    def forward(self, state):
        return self.actor.get_action(state)

    def learn(self, states, actions, rewards):
        '''
        Takes in three arguments each of which is a list with equal length. Each element in the list is a
        pytorch tensor with 1 row for every step in the episode, and the columns are state_dim, action_dim,
        and 1, respectively.
        '''

        # NOTES: implement the REINFORCE algorithm (HINT: check the slides/papers)

        returns = [compute_returns(rs, discount=self.discount) for rs in rewards]

        states, actions, returns = torch.cat(states), torch.cat(actions), torch.cat(returns)

        loss = -(Categorical(self.actor(states)).log_prob(actions) * (returns - self.baseline(states))).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        error = F.mse_loss(self.baseline(states).squeeze(), returns).detach()
        solve(states, returns, out=self.baseline)
        # error = F.mse_loss(self.baseline(states).squeeze(), returns).detach()

        return error.item()  # Returns a rough estimate of the error in the baseline (dont worry about this too much)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # NOTE: apply your value function network to get a value given this batch of states
        h1 = F.relu(self.fc_1.forward(state))
        h2 = F.relu(self.fc_2.forward(h1))

        return self.fc_3(h2)


class A3C(nn.Module):

    def __init__(self, state_dim, action_dim, discount=0.97, lr=1e-3, weight_decay=1e-4):
        super(A3C, self).__init__()
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        # NOTE: create an optimizer for the parameters of your actor (HINT: use the passed in lr and weight_decay args)
        # (HINT: the actor and critic have different objectives, so how many optimizers do you need?)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr, weight_decay=weight_decay)

        self.discount = discount

    def forward(self, state):
        return self.actor.get_action(state)

    def learn(self, states, actions, rewards):
        returns = [compute_returns(rs, discount=self.discount) for rs in rewards]

        states, actions, returns = torch.cat(states), torch.cat(actions), torch.cat(returns)

        # TODO: implement A3C (HINT: algorithm details found in A3C paper supplement)
        # (HINT2: the algorithm is actually very similar to REINFORCE, the only difference is now we have a critic, what might that do?)

        # actor optimize

        # IMPORTANT: use squeeze to avoid wrong broadcast
        critic_estimate = torch.squeeze(self.critic(states))
        advantage = returns - critic_estimate
        # scale loss by advantage
        actor_loss = -(Categorical(self.actor(states)).log_prob(actions) * advantage).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # critic optimize

        # Do another calculation because the grad won't work otherwise
        critic_estimate = torch.squeeze(self.critic(states))
        advantage = returns - critic_estimate
        critic_loss = advantage.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

#######################################
# Random Agent

class RandomAgent(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(RandomAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, state):
        return torch.randint(self.action_dim, size=(1,1))

    def learn(self, states, actions, rewards):
        pass
