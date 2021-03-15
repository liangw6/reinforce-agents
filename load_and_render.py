from agents import *
import sys

if len(sys.argv) < 2:
    print("Usage: python load_and_render saved_model.pt")
    sys.exit(1)

env_name = 'LunarLander-v2'
e = Pytorch_Gym_Env(env_name)
state_dim = e.observation_space.shape[0]
action_dim = e.action_space.n


agent = A3C(state_dim, action_dim)
agent.load_state_dict(torch.load(sys.argv[1]))
gen = Generator(e, agent)

agent.eval()
zip(*[gen(render=True) for _ in range(10)])