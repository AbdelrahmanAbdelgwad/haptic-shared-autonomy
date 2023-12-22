import sys
from time import time
from datetime import date
import os
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from gym import wrappers
from gym.envs.box2d.car_racing import CarRacingSharedStablebaselines3
from gym.envs.box2d.car_racing import CarRacingSharedStablebaselines3
from stable_baselines3.dqn_copilot.policies import (
    MultiInputPolicyCopilot,
    CnnPolicyCopilot,
)
from stable_baselines3.dqn_copilot.dqn import DQNCopilot
from stable_baselines3.common.callbacks import CallbackList
from haptic.callbacks.car_racing_callbacks import (
    SaveBestModelCallback,
    PeriodicSaveModelCallback,
)
from haptic.classic_control.pid_car_racing import pid, find_error

# Env Params
GRAYSCALE = 1  # 0 for RGB, 1 for Grayscale
SHOW_INFO_PANEL = 1  # 0 for no info panel, 1 for info panel
DISCRITIZED_ACTIONS = "smooth_steering"  # n_actions = 11
NUM_TRACKS = 1  # 1 for simple track, 2 for complex track
NUM_LANES = 1  # 1 for no lane changes, 2 for lane changes
NUM_LANES_CHANGES = 4  # DO NOT CHANGE
MAX_TIME_OUT = 2  # number of seconds out of track before episode ends
FRAMES_PER_STATE = 4  # number of frames to stack for each state

# Pilot Params
RANDOM_ACTION_PROB = (
    0.3  # probability of taking a random action if the pilot type is noisy
)
LAGGY_PILOT_FREQ = (
    3  # number of states to skip before taking an action if the pilot type is laggy
)

# Testing Params
NO_EPISODES = 1  # number of episodes to test for
MAX_EPISODE_TIMESTEPS = 1000  # number of timesteps per episode
PILOT_TYPES = [
    "human_keyboard",
    "none_pilot",
]  # ["none_pilot","laggy_pilot", "noisy_pilot", "optimal_pilot", "human_keyboard"]
ALPHA_SCHEDULE = [0.2, 0.6, 1]  # [0.0,..., 0.5,..., 1.0]


# Training Params
LOAD_MODEL = True  # True if you want to load a model
MODEL_NAME = "/home/mtr-pbl/haptic/haptic-shared-autonomy/best_model.zip"  # Name of the copilot model to load if the above is True

TIME_STEPS = 800_000  # number of timesteps to train for
LOG_INTERVAL = 20  # number of timesteps between each log
BUFFER_SIZE = 60_000  # size of the replay buffer
EVAL_FREQ = 500  # number of timesteps between each evaluation
RENDER_EVAL = False  # True if you want to render the evaluation
OUTPUT_PATH = "./training_folder"  # path to save the training folder
MODEL_SAVE_FREQ = 30_000  # number of timesteps between each model save


# PID Params
Kp = 0.02
Ki = 0.03
Kd = 0.2


def main():
    # Parsing command line args
    mode = sys.argv[1]  # train or test
    policy_type = sys.argv[2]  # Multi or Cnn
    pilot_path = sys.argv[3]  # path to load the pilot model
    if mode == "train":
        pilot_type = sys.argv[
            4
        ]  # none_pilot, laggy_pilot, noisy_pilot, optimal_pilot, human_keyboard
        copilot_path = sys.argv[5]  # path to save the copilot model
    elif mode == "test":
        copilot_path = sys.argv[4]  # path to load the copilot model

    # main code
    if mode == "train":
        today = date.today()
        date_str = today.strftime("%b-%d-%Y")
        train_folder_output_path = f"{OUTPUT_PATH}/{date_str}_{time()}"
        os.makedirs(train_folder_output_path)
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
            obs_space="frames",
            auto_render=False,
            scenario="train",
        )
        if policy_type == "Multi":
            if LOAD_MODEL:
                # Here we are loading the trained model and using its weights to initialize the new model
                # This is done to ensure that the weights are initialized to the same values as the trained model
                # However, we are free to change the hyperparameters of the new model
                trained_model = DQNCopilot.load(MODEL_NAME)
                state_dict = trained_model.q_net.state_dict()
                model = DQNCopilot(
                    MultiInputPolicyCopilot,
                    env,
                    buffer_size=BUFFER_SIZE,
                    verbose=1,
                    device="cuda",
                )
                model.q_net.load_state_dict(state_dict)
                model.q_net_target.load_state_dict(state_dict)
            else:
                model = DQNCopilot(
                    CnnPolicyCopilot,
                    env,
                    buffer_size=BUFFER_SIZE,
                    verbose=1,
                    device="cuda",
                )

        elif policy_type == "Cnn":
            if LOAD_MODEL:
                trained_model = DQNCopilot.load(MODEL_NAME)
                state_dict = trained_model.q_net.state_dict()
                model = DQNCopilot(
                    CnnPolicyCopilot,
                    env,
                    buffer_size=BUFFER_SIZE,
                    verbose=1,
                    device="cuda",
                    learning_rate = 0.00001,
                    exploration_fraction = 0.35,
                    exploration_initial_eps=0.05,
                    exploration_final_eps = 0.01,
                    tensorboard_log="./DQN_Copilot_tensorboard_trial/"
                )
                model.q_net.load_state_dict(state_dict)
                model.q_net_target.load_state_dict(state_dict)
            else:
                model = DQNCopilot(
                    CnnPolicyCopilot,
                    env,
                    buffer_size=BUFFER_SIZE,
                    verbose=1,
                    device="cuda",
                    elearning_rate = 0.00001,
                    exploration_fraction = 0.35,
                    exploration_initial_eps=0.05,
                    exploration_final_eps = 0.01,
                    tensorboard_log="./DQN_Copilot_tensorboard_trial/"
                )

        save_best_model_callback = SaveBestModelCallback(
            eval_env=env,
            n_eval_episodes=1,
            logpath=f"{train_folder_output_path}/logs",
            savepath=f"{train_folder_output_path}/best_model",
            eval_frequency=EVAL_FREQ,
            verbose=1,
            render=RENDER_EVAL,
        )
        save_model_callback = PeriodicSaveModelCallback(
            save_frequency=MODEL_SAVE_FREQ,
            save_path=f"{train_folder_output_path}/models",
        )
        callbacks = CallbackList([save_best_model_callback, save_model_callback])
        comment = input(
            "If you like to add a comment for the training add it here please: \n"
        )
        t1 = time()
        model.learn(
            total_timesteps=TIME_STEPS, log_interval=LOG_INTERVAL, callback=callbacks
        )
        model.save(f"{train_folder_output_path}/{copilot_path}")
        t2 = time()

        # Create a .txt file to save training summary
        summary_file_path = os.path.join(
            train_folder_output_path, "training_summary.txt"
        )
        with open(summary_file_path, "w") as summary_file:
            # Write general parameters
            summary_file.write("General Parameters:\n")
            summary_file.write(f"Mode: {mode}\n")
            summary_file.write(f"Policy Type: {policy_type}\n")
            summary_file.write(f"Pilot Path: {pilot_path}\n")

            # Write environment parameters
            summary_file.write("\nEnvironment Parameters:\n")
            summary_file.write(f"GRAYSCALE: {GRAYSCALE}\n")
            summary_file.write(f"SHOW_INFO_PANEL: {SHOW_INFO_PANEL}\n")
            summary_file.write(f"DISCRITIZED_ACTIONS: {DISCRITIZED_ACTIONS}\n")
            summary_file.write(f"NUM_TRACKS: {NUM_TRACKS}\n")
            summary_file.write(f"NUM_LANES: {NUM_LANES}\n")
            summary_file.write(f"NUM_LANES_CHANGES: {NUM_LANES_CHANGES}\n")
            summary_file.write(f"MAX_TIME_OUT: {MAX_TIME_OUT}\n")
            summary_file.write(f"FRAMES_PER_STATE: {FRAMES_PER_STATE}\n")

            # Write pilot parameters
            summary_file.write("\nPilot Parameters:\n")
            summary_file.write(f"RANDOM_ACTION_PROB: {RANDOM_ACTION_PROB}\n")
            summary_file.write(f"LAGGY_PILOT_FREQ: {LAGGY_PILOT_FREQ}\n")

            # Write training parameters
            summary_file.write("\nTraining Parameters:\n")
            summary_file.write(f"LOAD_MODEL: {LOAD_MODEL}\n")
            summary_file.write(f"MODEL_NAME: {MODEL_NAME}\n")
            summary_file.write(f"TIME_STEPS: {TIME_STEPS}\n")
            summary_file.write(f"LOG_INTERVAL: {LOG_INTERVAL}\n")
            summary_file.write(f"BUFFER_SIZE: {BUFFER_SIZE}\n")
            summary_file.write(f"EVAL_FREQ: {EVAL_FREQ}\n")
            summary_file.write(f"RENDER_EVAL: {RENDER_EVAL}\n")
            summary_file.write(f"Comment from the user: {comment}")

        print(f"Training summary saved at {summary_file_path}")
        print(f"Total time taken: {t2-t1} seconds")

    elif mode == "test":
        total_rewards = {}
        episode_rewards = {}
        episode_reward_list = []
        for alpha in ALPHA_SCHEDULE:
            episode_rewards[f"alpha_{alpha}"] = {}
            total_rewards[f"alpha_{alpha}"] = {}
        if sys.argv[2] == "Multi":
            for alpha in ALPHA_SCHEDULE:
                file_path = "./src/haptic/alpha.txt"
                with open(file_path, "w") as file:
                    file.write(str(alpha))
                for pilot_type in PILOT_TYPES:
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
                        obs_space="dict",
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
                            action, _ = model.predict(observation,device='cuda:1')
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
            for alpha in ALPHA_SCHEDULE:
                file_path = "./src/haptic/alpha.txt"
                with open(file_path, "w") as file:
                    file.write(str(alpha))
                for pilot_type in PILOT_TYPES:
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
                        obs_space="frames",
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
                            action, _ = model.predict(observation,device='cuda:1')
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

        elif sys.argv[2] == "PID":
            for alpha in ALPHA_SCHEDULE:
                for pilot_type in PILOT_TYPES:
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
                        obs_space="frames",
                        display=f"PID: Kp = {Kp}, Ki = {Ki}, Kd = {Kd}",
                    )
                    env = wrappers.Monitor(
                        env,
                        f"./videos/PID_Kp={Kp}_Ki={Ki}_Kd={Kd}_video/",
                        force=True,
                    )
                    episode_timesteps = 0
                    done = False
                    episode_reward = 0
                    total_timesteps = 0
                    total_reward = 0
                    observation = env.reset()
                    previous_error = 0
                    for episode in range(NO_EPISODES):
                        while not done:
                            episode_timesteps += 1
                            total_timesteps += 1
                            env.render()
                            error = find_error(observation, previous_error)
                            steering = pid(error, previous_error, Kp, Ki, Kd)
                            steering = steering + np.random.normal(0, 0.2)
                            action = [steering, 0.3, 0.05]
                            observation, reward, done, info = env.step(action)
                            episode_reward += reward
                            total_reward += reward
                            previous_error = error
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

        for alpha in ALPHA_SCHEDULE:
            total_rewards_alpha = [
                total_rewards[f"alpha_{alpha}"][pilot_type]
                for pilot_type in PILOT_TYPES
            ]
            plt.figure()
            plt.bar(PILOT_TYPES, total_rewards_alpha)
            plt.xlabel("Pilot Type")
            plt.ylabel("Total Reward")
            plt.title(f"Total Rewards for Alpha = {alpha}")
            plt.savefig(f"./charts/total_rewards_alpha_{alpha}.png")

            episode_rewards_alpha = [
                episode_rewards[f"alpha_{alpha}"][pilot_type]
                for pilot_type in PILOT_TYPES
            ]
            plt.figure()
            plt.bar(PILOT_TYPES, episode_rewards_alpha)
            plt.xlabel("Pilot Type")
            plt.ylabel("Episode Reward")
            plt.title(f"Episode Rewards for Alpha = {alpha}")
            plt.savefig(f"./charts/episode_rewards_alpha_{alpha}.png")

        for pilot_type in PILOT_TYPES:
            episode_rewards_pilot = [
                episode_rewards[f"alpha_{alpha}"][pilot_type]
                for alpha in ALPHA_SCHEDULE
            ]
            plt.figure()
            plt.bar([str(alpha) for alpha in ALPHA_SCHEDULE], episode_rewards_pilot)
            plt.xlabel("Alpha Value")
            plt.ylabel("Episode Reward")
            plt.title(f"Episode Rewards for Pilot Type = {pilot_type}")
            plt.savefig(f"./charts/episode_rewards_{pilot_type}.png")

            total_rewards_pilot = [
                total_rewards[f"alpha_{alpha}"][pilot_type] for alpha in ALPHA_SCHEDULE
            ]
            plt.figure()
            plt.bar([str(alpha) for alpha in ALPHA_SCHEDULE], total_rewards_pilot)
            plt.xlabel("Alpha Value")
            plt.ylabel("Total Reward")
            plt.title(f"Total Rewards for Pilot Type = {pilot_type}")
            plt.savefig(f"./charts/total_rewards_{pilot_type}.png")


if __name__ == "__main__":
    t1 = time()
    main()
