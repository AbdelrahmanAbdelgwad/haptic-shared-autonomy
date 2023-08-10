import sys
from time import time
from gym import wrappers
from gym.envs.box2d.car_racing import CarRacingSharedStablebaselines3
from gym.envs.box2d.car_racing import CarRacingSharedStablebaselines3
from stable_baselines3.dqn_copilot.policies import (
    MultiInputPolicyCopilot,
    CnnPolicyCopilot,
)
from stable_baselines3.dqn_copilot.dqn import DQNCopilot


if __name__ == "__main__":
    if sys.argv[1] == "train":
        if sys.argv[2] == "Multi":
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
                pilot_type="optimal_pilot",
                random_action_prob=0.2,
                laggy_pilot_freq=4,
                use_dict_obs_space=True,
            )

            model = DQNCopilot(
                MultiInputPolicyCopilot, env, buffer_size=100_000, verbose=1
            )
            model.learn(total_timesteps=1000, log_interval=4)
            model.save("copilot_stablebaselines3_Multi")

        elif sys.argv[2] == "Cnn":
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
                pilot_type="optimal_pilot",
                random_action_prob=0.2,
                laggy_pilot_freq=4,
                use_dict_obs_space=False,
            )

            model = DQNCopilot(CnnPolicyCopilot, env, buffer_size=100_000, verbose=1)
            model.learn(total_timesteps=1000, log_interval=4)
            model.save("copilot_stablebaselines3_Cnn")

    elif sys.argv[1] == "test":
        # pilot_list = ["none", "laggy", "noisy", "optimal"]
        pilot_list = ["optimal"]
        NO_EPISODES = 1
        MAX_EPISODE_TIMESTEPS = 1000
        if sys.argv[2] == "Multi":
            t1 = time()
            for pilot in pilot_list:
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
                    pilot_type="optimal_pilot",
                    random_action_prob=0.2,
                    laggy_pilot_freq=4,
                    use_dict_obs_space=True,
                )
                env = wrappers.Monitor(
                    env, f"./copilot_{pilot}_pilot_video/", force=True
                )
                model = DQNCopilot.load("copilot_stablebaselines3_Multi")
                # model = DQN_copilot(CnnPolicy, env=env, buffer_size=5000)
                # model = model.load("dqn_car")
                episode_timesteps = 0
                done = False
                episode_reward = 0
                avg_reward = 0
                total_timesteps = 0
                observation = env.reset()
                for episode in range(NO_EPISODES):
                    while not done:
                        episode_timesteps += 1
                        total_timesteps += 1
                        env.render()
                        action, _ = model.predict(observation)
                        print(action)
                        observation, reward, done, info = env.step(action)
                        episode_reward += reward
                        if done:
                            env.reset()
                            done = False
                        if episode_timesteps % MAX_EPISODE_TIMESTEPS == 0:
                            episode_timesteps = 0
                            break
                env.close()

        elif sys.argv[2] == "Cnn":
            t1 = time()
            for pilot in pilot_list:
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
                    pilot_type="optimal_pilot",
                    random_action_prob=0.2,
                    laggy_pilot_freq=4,
                    use_dict_obs_space=False,
                )
                env = wrappers.Monitor(
                    env, f"./copilot_{pilot}_pilot_video/", force=True
                )
                model = DQNCopilot.load("copilot_stablebaselines3_Cnn")
                # model = DQN_copilot(CnnPolicy, env=env, buffer_size=5000)
                # model = model.load("dqn_car")
                episode_timesteps = 0
                done = False
                episode_reward = 0
                avg_reward = 0
                total_timesteps = 0
                observation = env.reset()
                for episode in range(NO_EPISODES):
                    while not done:
                        episode_timesteps += 1
                        total_timesteps += 1
                        env.render()
                        action, _ = model.predict(observation)
                        print(action)
                        observation, reward, done, info = env.step(action)
                        episode_reward += reward
                        if done:
                            env.reset()
                            done = False
                        if episode_timesteps % MAX_EPISODE_TIMESTEPS == 0:
                            episode_timesteps = 0
                            break
                env.close()
