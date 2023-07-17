from stable_baselines3.dqn.dqn import DQN
from haptic.gym.envs.box2d.car_racing import CarRacingSharedStablebaselines3
from haptic.gym import wrappers
env = CarRacingSharedStablebaselines3(
    allow_reverse=False,
    grayscale=1,
    show_info_panel=1,
    discretize_actions="smooth_steering",  # n_actions = 11
    num_tracks=2,
    num_lanes=2,
    num_lanes_changes=4,
    max_time_out=5,
    frames_per_state=4,
    pilot="trials/models/FINAL_MODEL_SMOOTH_STEERING_CAR",
    pilot_type="laggy_pilot",
    random_action_prob = 0.2,
    laggy_pilot_freq = 4
)
env = wrappers.Monitor(env, "./copilot_with_50%_random_pilot_video/", force=True)
model = DQN.load("copilot_stablebaselines3")
t = 0
done = False
episode_reward = 0
observation = env.reset()
# Notice that episodes here are very small due to the way that the environment is structured
for episode in range(1000):
    while not done:
        t += 1
        env.render()
        action, _ = model.predict(observation)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
        break
    print(episode_reward)
    episode_reward = 0
env.close()
