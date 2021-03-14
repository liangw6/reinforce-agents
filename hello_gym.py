import gym


def run_random_agent(env_name='CartPole-v0'):
    env = gym.make(env_name)
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print('at t', t)
            print('\t observation:', observation)
            print('\t action space:', env.action_space)

            # 4 actions: do nothing, left, center, right
            action = env.action_space.sample()
            if action == 1 or action == 3:
                # don't fire side engines
                action = 0
            observation, reward, done, info = env.step(action)

            print('\t action', action)
            print('\t reward', reward)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


run_random_agent('LunarLander-v2')