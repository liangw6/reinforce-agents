from agents import *

env_name = 'LunarLander-v2'
e = Pytorch_Gym_Env(env_name)
state_dim = e.observation_space.shape[0]
action_dim = e.action_space.n


agent = A3C(state_dim, action_dim)
agent.load_state_dict(torch.load('saved_models/actor_only_3000.pt'))
gen = Generator(e, agent)

agent.eval()
zip(*[gen(render=True) for _ in range(10)])