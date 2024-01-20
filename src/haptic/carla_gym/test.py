# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import gym
import os
import csv
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from haptic.carla_gym.carla_env import CarlaEnv
from haptic.utils.carla_utils import *


# # ==============================================================================
# # -- Global Parameters ---------------------------------------------------------
# # ==============================================================================

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set the path for the model
# model_path = "/home/mtr-pbl/haptic/data_sets/best_model_1152_carla_18.pth"
model_path = "/home/mtr-pbl/haptic/data_sets/1152/best_model_1152_p_10.pth"

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

evaluate_model = False


# initiate class for the model
class NetworkNvidia(nn.Module):
    """NVIDIA model used in the paper."""

    def __init__(self):
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
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1),
        )

    def forward(self, input):
        """Forward pass."""
        input = input.view(-1, 3, 66, 200)
        output = self.conv_layers(input)
        # print(output.shape)
        output = output.view(-1, 1152)
        output = self.linear_layers(output)
        return output


# initiate class for the model
model = NetworkNvidia().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()


# ==============================================================================
# -- Main ----------------------------------------------------------------------
# ==============================================================================


def main():
    env = None
    # parameters for the carla_gym environment
    params = {
        "max_time_episode": 1000,  # maximum timesteps per episode
        "obs_size": [66, 200],  # observation (image) size[height,width]
        "min_speed": 10,  # desired minimum eg vehicle speed (Km/Hr)
        "max_speed": 15,  # desired maximum eg vehicle speed (Km/Hr)
        "discrete": False,  # whether to use discrete control space
        "discrete_steer": [-0.2, 0.0, 0.2],  # discrete value of steering angles
        "continuous_steer_range": [-1, 1],  # continuous steering angle range
        "scenario": "test",
    }

    try:
        # Set carla-gym environment
        # env = gym.make("Carla-v0", params=params)
        env = CarlaEnv(params=params)
        episodes = 5
        frames = 1
        agent_angles = []
        model_angles = []

        for i in range(1, episodes + 1):
            obs, info = env.reset()  # MOD: added info extraction

            # if i < 2:
            #     cv2.imwrite("a.jpg", obs["camera"])

            done = False
            score = 0

            while not done:
                if not evaluate_model:
                    model_steer = predict_steering_angle(
                        obs,
                        model,
                        device,
                    )  # random action selection
                    print("Steering Angle: {:.3f}".format(model_steer))

                    action = [model_steer]
                    obs, reward, done, _, _ = env.step(action)

                    score += reward

                else:
                    # Model Commands
                    model_steer = predict_steering_angle(obs)  # random action selection
                    # model_steer = 0
                    model_angles.append(model_steer)

                    # Agent (Autopilot) Commands
                    agent_ctrl = env.agent.run_step()
                    agent_steer = agent_ctrl.steer
                    agent_angles.append(agent_steer)

                    action = [agent_steer]
                    obs, reward, done, _, info = env.step(action)

                    score += reward

                    frames += 1

                # print(f"Collision Histroy: {info['Collision']}")
                # print(f"Lane INvasion Timestamps: {info['LaneInv']}")

            # print(f"Final Score: {score}")

    except KeyboardInterrupt:
        generate_model_evaluation(frames, agent_angles, model_angles, pkg_dir)

        if env != None:
            env.destroy_actors()
        print("\n>>> Cancelled by user. Bye!\n")

    finally:
        generate_model_evaluation(frames, agent_angles, model_angles, pkg_dir)
        if env != None:
            env.destroy_actors()
        print("\n>>> Cancelled by user. Bye!\n")


if __name__ == "__main__":
    main()
