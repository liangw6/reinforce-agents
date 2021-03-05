import gym


def cartPoleRun():
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print('at t', t)
            print('\t observation:',observation)
            print('\t action space:', env.action_space)

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            print('\t action', action)
            print('\t reward', reward)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


def lunarLander():
    env = gym.make('LunarLander-v2')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


# lunarLander()
cartPoleRun()