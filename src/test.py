import haptic.gym as gym
from haptic.gym.envs.box2d.car_racing import CarRacing
from stable_baselines3 import DQN

env = CarRacing(
    allow_reverse=False,
    grayscale=1,
    show_info_panel=1,
    discretize_actions="soft",
    num_tracks=2,
    num_lanes=2,
    num_lanes_changes=4,
    max_time_out=0,
    frames_per_state=4,
)
# Uncomment following line to save video of our Agent interacting in this environment
# This can be used for debugging and studying how our agent is performing
env = gym.wrappers.Monitor(env, "./video/", force=True)
model = DQN.load("DQN_model_hard_actions")
t = 0
done = False
episode_reward = 0
observation = env.reset()
# Notice that episodes here are very small due to the way that the environment is structured
for episode in range(10000):
    while not done:
        t += 1
        env.render()
        #    print(observation)
        action, _ = model.predict(observation)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
        break
    print(episode_reward)
    episode_reward = 0
env.close()
