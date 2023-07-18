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
    pilot_type="noisy_pilot",
    random_action_prob=0.2,
    laggy_pilot_freq=4,
)
env = wrappers.Monitor(env, "./copilot_noisy_pilot_video/", force=True)
model = DQN.load("copilot_stablebaselines3")
episode_timesteps = 0
done = False
episode_reward = 0
NO_EPISODES = 10
MAX_EPISODE_TIMESTEPS = 1000
observation = env.reset()
avg_reward = 0
for episode in range(NO_EPISODES):
    while not done:
        episode_timesteps += 1
        env.render()
        action, _ = model.predict(observation)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            env.reset()
            done = False
        episode_reward = 0
        if episode_timesteps % MAX_EPISODE_TIMESTEPS == 0:
            episode_timesteps = 0
            break
    avg_reward += episode_reward
print("Average Reward =", avg_reward)
env.close()
