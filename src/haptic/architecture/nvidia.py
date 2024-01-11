"""
CNN model architecture.

taken from: Zhenye Na - https://github.com/Zhenye-Na
reference: "End to End Learning for Self-Driving Cars", arXiv:1604.07316
"""
from time import time
from datetime import date
import os

import gym
from gym.envs.box2d.car_racing import CarRacing

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ddpg.ddpg import DDPG

from haptic.callbacks.car_racing_callbacks import (
    SaveBestModelCallback,
    PeriodicSaveModelCallback,
)


class NetworkNvidia(nn.Module):
    """NVIDIA model used in the paper."""

    def __init__(self, features_dim: int = 50):
        """Initialize NVIDIA model.

        NVIDIA model used
            Image normalization to avoid saturation and make gradients work better.
            Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Drop out (0.5)
            Fully connected: neurons: 100, activation: ELU
            Fully connected: neurons: 50, activation: ELU
            Fully connected: neurons: 10, activation: ELU
            Fully connected: neurons: 1 (output)

        the convolution layers are meant to handle feature engineering.
        the fully connected layer for predicting the steering angle.
        the elu activation function is for taking care of vanishing gradient problem.
        """
        super(NetworkNvidia, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=1152, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=features_dim),
            nn.ELU(),
            # nn.Linear(in_features=50, out_features=10),
            # nn.Linear(in_features=10, out_features=1),
        )

    def forward(self, input):
        """Forward pass."""
        print(input.shape)
        output = self.conv_layers(input)
        print(output.shape)
        # output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 50):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]

        nvidia_net = NetworkNvidia(features_dim)
        self.cnn = nvidia_net.conv_layers
        self.linear = nvidia_net.linear_layers

    def forward(self, observations: th.Tensor) -> th.Tensor:
        output = self.cnn(observations)
        flatten_output = th.flatten(output, start_dim=1)
        output = self.linear(flatten_output)
        return output


if __name__ == "__main__":
    # Env Params
    GRAYSCALE = 0  # 0 for RGB, 1 for Grayscale
    SHOW_INFO_PANEL = 1  # 0 for no info panel, 1 for info panel
    DISCRITIZED_ACTIONS = None
    NUM_TRACKS = 1  # 1 for simple track, 2 for complex track
    NUM_LANES = 1  # 1 for no lane changes, 2 for lane changes
    NUM_LANES_CHANGES = 4  # DO NOT CHANGE
    MAX_TIME_OUT = 2  # number of seconds out of track before episode ends
    FRAMES_PER_STATE = 4  # number of frames to stack for each state
    NVIDIA = True  # True for NVIDIA model, False for normal model with 96,96,3 input
    BUFFER_SIZE = 60_000
    TOTAL_TIMESTEPS = 500_000

    EVAL_FREQ = 50_000  # number of timesteps between each evaluation
    RENDER_EVAL = False  # True if you want to render the evaluation
    OUTPUT_PATH = "./training_folder"  # path to save the training folder
    MODEL_SAVE_FREQ = 50_000  # number of timesteps between each model save

    today = date.today()
    date_str = today.strftime("%b-%d-%Y")
    train_folder_output_path = f"{OUTPUT_PATH}/{date_str}_{time()}"
    os.makedirs(train_folder_output_path)

    env = CarRacing(
        allow_reverse=False,
        grayscale=GRAYSCALE,
        show_info_panel=SHOW_INFO_PANEL,
        discretize_actions=DISCRITIZED_ACTIONS,
        num_tracks=NUM_TRACKS,
        num_lanes=NUM_LANES,
        num_lanes_changes=NUM_LANES_CHANGES,
        max_time_out=MAX_TIME_OUT,
        frames_per_state=FRAMES_PER_STATE,
        nvidia=NVIDIA,
    )

    # Custom actor architecture with two layers of 50 and 10 units respectively
    # Custom critic architecture with two layers of 50, 100, 50, and 10 units respectively
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        net_arch=dict(pi=[50, 10], qf=[50, 100, 50, 10]),
    )

    # Create the agent
    model = DDPG(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        buffer_size=BUFFER_SIZE,
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
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    model.save("ddpg_car_racing")
