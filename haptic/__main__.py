def add(x, y):
    return x + y


if __name__ == "__main__":
    import gym
    from stable_baselines3 import PPO

    env = gym.make("CarRacing-v0")

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    model.save("ppo_carracing")

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()
