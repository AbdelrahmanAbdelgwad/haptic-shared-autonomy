import sys
from time import time
import os
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers
from gym.envs.box2d.car_racing import CarRacingSharedStablebaselines3
from gym.envs.box2d.car_racing import CarRacingSharedStablebaselines3
from stable_baselines3.dqn_copilot.policies import (
    MultiInputPolicyCopilot,
    CnnPolicyCopilot,
)
from stable_baselines3.dqn_copilot.dqn import DQNCopilot

# Copilot naming convention:
# copilot_<training_timesteps>_<alpha_used>_<pilot_type>_<random_action_prob>_<laggy_pilot_freq>_<network_type>
# if some property is not used due to the pilot type then follow it with an x

# Env Params
GRAYSCALE = 1
SHOW_INFO_PANEL = 1
DISCRITIZED_ACTIONS = "smooth_steering"  # n_actions = 11
NUM_TRACKS = 2
NUM_LANES = 2
NUM_LANES_CHANGES = 4
MAX_TIME_OUT = 5
FRAMES_PER_STATE = 4

# Pilot Params
RANDOM_ACTION_PROB = 0.3
LAGGY_PILOT_FREQ = 4

# Testing Params
NO_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 500


def main():
    # Parsing command line args
    mode = sys.argv[1]
    policy_type = sys.argv[2]
    pilot_path = sys.argv[3]
    if mode == "train":
        pilot_type = sys.argv[4]
        copilot_path = sys.argv[5]
    elif mode == "test":
        copilot_path = sys.argv[4]

    # main code
    if mode == "train":
        if policy_type == "Multi":
            env = CarRacingSharedStablebaselines3(
                allow_reverse=False,
                grayscale=GRAYSCALE,
                show_info_panel=SHOW_INFO_PANEL,
                discretize_actions=DISCRITIZED_ACTIONS,
                num_tracks=NUM_TRACKS,
                num_lanes=NUM_LANES,
                num_lanes_changes=NUM_LANES_CHANGES,
                max_time_out=MAX_TIME_OUT,
                frames_per_state=FRAMES_PER_STATE,
                pilot=f"{pilot_path}",
                pilot_type=f"{pilot_type}",
                random_action_prob=RANDOM_ACTION_PROB,
                laggy_pilot_freq=LAGGY_PILOT_FREQ,
                use_dict_obs_space=True,
            )

            model = DQNCopilot(
                MultiInputPolicyCopilot, env, buffer_size=100_000, verbose=1
            )
            model.learn(total_timesteps=1000_000, log_interval=4)
            model.save(f"{copilot_path}")

        elif policy_type == "Cnn":
            env = CarRacingSharedStablebaselines3(
                allow_reverse=False,
                grayscale=GRAYSCALE,
                show_info_panel=SHOW_INFO_PANEL,
                discretize_actions=DISCRITIZED_ACTIONS,
                num_tracks=NUM_TRACKS,
                num_lanes=NUM_LANES,
                num_lanes_changes=NUM_LANES_CHANGES,
                max_time_out=MAX_TIME_OUT,
                frames_per_state=FRAMES_PER_STATE,
                pilot=f"{pilot_path}",
                pilot_type=f"{pilot_type}",
                random_action_prob=RANDOM_ACTION_PROB,
                laggy_pilot_freq=LAGGY_PILOT_FREQ,
                use_dict_obs_space=False,
            )

            model = DQNCopilot(
                CnnPolicyCopilot,
                env,
                buffer_size=50_000,
                verbose=1,
            )
            model.learn(total_timesteps=1000_000, log_interval=4)
            model.save(f"{copilot_path}")

    elif mode == "test":
        # pilot_types = ["none_pilot", "laggy_pilot", "noisy_pilot", "optimal_pilot"]
        # alpha_schedule = [0, 0.3, 0.6, 1]
        pilot_types = ["human_keyboard", "optimal_pilot"]
        alpha_schedule = [1, 0.6, 0.3, 0]
        total_rewards = {}
        episode_rewards = {}
        episode_reward_list = []
        for alpha in alpha_schedule:
            episode_rewards[f"alpha_{alpha}"] = {}
            total_rewards[f"alpha_{alpha}"] = {}
        if sys.argv[2] == "Multi":
            for alpha in alpha_schedule:
                file_path = "/home/hydra/grad_project/haptic-shared-autonomy/src/haptic/alpha.txt"
                with open(file_path, "w") as file:
                    file.write(str(alpha))
                for pilot_type in pilot_types:
                    env = CarRacingSharedStablebaselines3(
                        allow_reverse=False,
                        grayscale=GRAYSCALE,
                        show_info_panel=SHOW_INFO_PANEL,
                        discretize_actions=DISCRITIZED_ACTIONS,
                        num_tracks=NUM_TRACKS,
                        num_lanes=NUM_LANES,
                        num_lanes_changes=NUM_LANES_CHANGES,
                        max_time_out=MAX_TIME_OUT,
                        frames_per_state=FRAMES_PER_STATE,
                        pilot=f"{pilot_path}",
                        pilot_type=f"{pilot_type}",
                        random_action_prob=RANDOM_ACTION_PROB,
                        laggy_pilot_freq=LAGGY_PILOT_FREQ,
                        use_dict_obs_space=True,
                        display=f"copilot_alpha_{alpha}_{pilot_type}",
                    )
                    env = wrappers.Monitor(
                        env,
                        f"./videos/copilot_alpha_{alpha}_{pilot_type}_video/",
                        force=True,
                    )
                    model = DQNCopilot.load(f"{copilot_path}")
                    episode_timesteps = 0
                    done = False
                    episode_reward = 0
                    total_timesteps = 0
                    total_reward = 0
                    observation = env.reset()
                    for episode in range(NO_EPISODES):
                        while not done:
                            episode_timesteps += 1
                            total_timesteps += 1
                            env.render()
                            action, _ = model.predict(observation)
                            observation, reward, done, info = env.step(action)
                            episode_reward += reward
                            total_reward += reward
                            if done:
                                env.reset()
                                done = False
                            if episode_timesteps % MAX_EPISODE_TIMESTEPS == 0:
                                episode_reward_list.append(episode_reward)
                                episode_reward = 0
                                episode_timesteps = 0
                            if (
                                total_timesteps % (MAX_EPISODE_TIMESTEPS * NO_EPISODES)
                                == 0
                            ):
                                total_rewards[f"alpha_{alpha}"][
                                    f"{pilot_type}"
                                ] = total_reward
                                episode_rewards[f"alpha_{alpha}"][
                                    f"{pilot_type}"
                                ] = np.mean(episode_reward_list)
                                break
                    env.close()

        elif sys.argv[2] == "Cnn":
            for alpha in alpha_schedule:
                file_path = "/home/hydra/grad_project/haptic-shared-autonomy/src/haptic/alpha.txt"
                with open(file_path, "w") as file:
                    file.write(str(alpha))
                for pilot_type in pilot_types:
                    env = CarRacingSharedStablebaselines3(
                        allow_reverse=False,
                        grayscale=GRAYSCALE,
                        show_info_panel=SHOW_INFO_PANEL,
                        discretize_actions=DISCRITIZED_ACTIONS,
                        num_tracks=NUM_TRACKS,
                        num_lanes=NUM_LANES,
                        num_lanes_changes=NUM_LANES_CHANGES,
                        max_time_out=MAX_TIME_OUT,
                        frames_per_state=FRAMES_PER_STATE,
                        pilot=f"{pilot_path}",
                        pilot_type=f"{pilot_type}",
                        random_action_prob=RANDOM_ACTION_PROB,
                        laggy_pilot_freq=LAGGY_PILOT_FREQ,
                        use_dict_obs_space=False,
                        display=f"copilot_alpha_{alpha}_{pilot_type}",
                    )
                    env = wrappers.Monitor(
                        env,
                        f"./videos/copilot_alpha_{alpha}_{pilot_type}_video/",
                        force=True,
                    )
                    model = DQNCopilot.load(f"{copilot_path}")
                    episode_timesteps = 0
                    done = False
                    episode_reward = 0
                    total_timesteps = 0
                    total_reward = 0
                    observation = env.reset()
                    for episode in range(NO_EPISODES):
                        while not done:
                            episode_timesteps += 1
                            total_timesteps += 1
                            env.render()
                            action, _ = model.predict(observation)
                            observation, reward, done, info = env.step(action)
                            episode_reward += reward
                            total_reward += reward
                            if done:
                                env.reset()
                                done = False
                            if episode_timesteps % MAX_EPISODE_TIMESTEPS == 0:
                                episode_reward_list.append(episode_reward)
                                episode_reward = 0
                                episode_timesteps = 0
                            if (
                                total_timesteps % (MAX_EPISODE_TIMESTEPS * NO_EPISODES)
                                == 0
                            ):
                                total_rewards[f"alpha_{alpha}"][
                                    f"{pilot_type}"
                                ] = total_reward
                                episode_rewards[f"alpha_{alpha}"][
                                    f"{pilot_type}"
                                ] = np.mean(episode_reward_list)
                                break
                    env.close()

            # Create a directory for saving charts if it doesn't exist
            chart_dir = "./charts"
            if not os.path.exists(chart_dir):
                os.makedirs(chart_dir)

            for alpha in alpha_schedule:
                total_rewards_alpha = [
                    total_rewards[f"alpha_{alpha}"][pilot_type]
                    for pilot_type in pilot_types
                ]
                plt.figure()
                plt.bar(pilot_types, total_rewards_alpha)
                plt.xlabel("Pilot Type")
                plt.ylabel("Total Reward")
                plt.title(f"Total Rewards for Alpha = {alpha}")
                plt.savefig(f"./charts/total_rewards_alpha_{alpha}.png")

                episode_rewards_alpha = [
                    episode_rewards[f"alpha_{alpha}"][pilot_type]
                    for pilot_type in pilot_types
                ]
                plt.figure()
                plt.bar(pilot_types, episode_rewards_alpha)
                plt.xlabel("Pilot Type")
                plt.ylabel("Episode Reward")
                plt.title(f"Episode Rewards for Alpha = {alpha}")
                plt.savefig(f"./charts/episode_rewards_alpha_{alpha}.png")

            for pilot_type in pilot_types:
                episode_rewards_pilot = [
                    episode_rewards[f"alpha_{alpha}"][pilot_type]
                    for alpha in alpha_schedule
                ]
                plt.figure()
                plt.bar([str(alpha) for alpha in alpha_schedule], episode_rewards_pilot)
                plt.xlabel("Alpha Value")
                plt.ylabel("Episode Reward")
                plt.title(f"Episode Rewards for Pilot Type = {pilot_type}")
                plt.savefig(f"./charts/episode_rewards_{pilot_type}.png")

                total_rewards_pilot = [
                    total_rewards[f"alpha_{alpha}"][pilot_type]
                    for alpha in alpha_schedule
                ]
                plt.figure()
                plt.bar([str(alpha) for alpha in alpha_schedule], total_rewards_pilot)
                plt.xlabel("Alpha Value")
                plt.ylabel("Total Reward")
                plt.title(f"Total Rewards for Pilot Type = {pilot_type}")
                plt.savefig(f"./charts/total_rewards_{pilot_type}.png")


if __name__ == "__main__":
    main()
