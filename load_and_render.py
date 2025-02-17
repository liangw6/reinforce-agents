from agents import *
import sys
import argparse

parser = argparse.ArgumentParser(description='load, play and render a saved pytorch RL agent')
parser.add_argument('saved_model', help='path to a saved model')
parser.add_argument('--N', help='Number of Episodes', type=int, default=1)
parser.add_argument('--env', choices=['lander', 'cartpole'], help='the environment for openAI gym', required=True)
parser.add_argument('--output', help='if specified, the output directory to save video to')
parser.add_argument('--eval', action='store_true', help='If set, run N = 10 and print average reward (no render)')


args = vars(parser.parse_args())

# system param
N = args['N']
saved_model_params = args['saved_model']
if args['env'] == 'lander':
    env_name = 'LunarLander-v2'
elif args['env'] == 'cartpole':
    env_name = 'CartPole-v1'
else:
    print('unknown environment', args['env'])
    sys.exit(1)
output_dir = args['output']

# starting up the gym
e = Pytorch_Gym_Env(env_name, output=output_dir)
state_dim = e.observation_space.shape[0]
action_dim = e.action_space.n
# start up agent
if saved_model_params == 'random':
    # random agent
    agent = RandomAgent(state_dim, action_dim)
else:
    # otherwise, A3C
    agent = A3C(state_dim, action_dim)
    agent.load_state_dict(torch.load(saved_model_params))

gen = Generator(e, agent)
agent.eval()

# Run and Render!
if args['eval']:
    N = 100 # use 10 for lunarlander so it's faster
    states, actions, rewards = zip(*[gen(render=False) for _ in range(N)])
else:
    states, actions, rewards = zip(*[gen(render=True) for _ in range(N)])

reward = sum([r.sum() for r in rewards]) / N

print('Average Reward: {} for {} episodes'.format(reward, N))

if output_dir is not None:
    e.close()

