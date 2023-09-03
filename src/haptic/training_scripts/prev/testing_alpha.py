import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.dqn_copilot import CnnPolicyCopilot
from stable_baselines3 import DQN
from stable_baselines3 import DQNCopilot
from gym.envs.box2d.car_racing import CarRacingSharedStablebaselines3, CarRacing
from gym import wrappers
from time import time

import numpy as np
import matplotlib.pyplot as plt


def display_grayscale_images(arrays):
    num_images = 5
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))

    for i in range(num_images):
        array = arrays[:, :, i]
        axes[i].imshow(array, cmap="gray")
        axes[i].axis("off")

    plt.savefig(f"frames")


import numpy as np


def save_numpy_array(array, file_path):
    np.save(file_path, array)


def load_numpy_array(file_path):
    return np.load(file_path)


ALPHA = 1
RANDOM_ACTION_PROB = 0.2
LAG_FREQ = 4
NO_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 2000

if __name__ == "__main__":
    t1 = time()
    false_counter = 0
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
        pilot_type=f"optimal_pilot",
        random_action_prob=RANDOM_ACTION_PROB,
        laggy_pilot_freq=LAG_FREQ,
    )
    copilot = DQNCopilot.load("copilot_stablebaselines3")
    pilot = DQN.load("trials/models/FINAL_MODEL_SMOOTH_STEERING_CAR")
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
            state = observation[:, :, 0:4]
            pi_action, _ = pilot.predict(state)
            print(pi_action)
            action, _ = copilot.predict(observation)
            # if pi_action != action:
            #     false_counter+=1
            #     print("increment")
            # if false_counter > 2:
            #     # print(observation.shape)
            #     display_grayscale_images(observation)
            #     # Save the numpy array to a file
            #     file_path = "saved_array.npy"
            #     save_numpy_array(observation[:,:,0:4], file_path)
            #     print("Numpy array saved successfully!")

            #     break
            observation, reward, done, info = env.step(pi_action)
            episode_reward += reward
            if done:
                env.reset()
                done = False
            if episode_timesteps % MAX_EPISODE_TIMESTEPS == 0:
                episode_timesteps = 0
                break
        avg_reward += episode_reward
    env.close()
    # print(false_counter)

    t2 = time()
    delta_t = (t2 - t1) / 60
    print(f"took {delta_t} minutes")

    # print(pilot.predict(load_numpy_array(file_path)))
