import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, "recordings")

    obs = env.reset()
    total_reward = 0.0

    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

        total_reward += reward

        if done:
            break

    print(f"Total reward is {total_reward}")
    env.close()
    env.env.close()