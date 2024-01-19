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


# # ==============================================================================
# # -- Global Functions ----------------------------------------------------------
# # ==============================================================================


def generate_model_evaluation(frames, agent_angles, model_angles):
    """
    Create CSV file of dataset labels (Steering Angles in Deg)"""
    frame_lst = []
    agent_lst = []
    model_lst = []
    err_lst = []
    acc_lst = []
    for i in range(frames):
        frame_lst.append(str(i + 1))  # Form Frames Column
        agent_lst.append("%.5f" % agent_angles[i])  # Format float to 5 decimals
        model_lst.append("%.5f" % model_angles[i])  # Format float to 5 decimals
        err = abs(agent_angles[i] - model_angles[i])  # Form Error Column
        err_lst.append("%.5f" % err)  # Format float to 5 decimals
        acc = round(abs(err * 100 / agent_angles[i]))
        acc_lst.append(acc)

    # Convert lists to numpy arrays
    frame_lst = np.array(frame_lst)
    agent_lst = np.array(agent_lst)
    model_lst = np.array(model_lst)
    err_lst = np.array(err_lst)
    acc_lst = np.array(acc_lst)

    # Concatenate numpy arrays for CSV data
    data = np.stack((frame_lst, agent_lst, model_lst, err_lst, acc_lst), axis=1)

    # Write data to CSV file
    csv_path = os.path.join(pkg_dir, "results.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    # Convert CSV file to excel
    excel_path = os.path.join(pkg_dir, "results.xlsx")
    csv_file = pd.read_csv(csv_path)
    csv_file.to_excel(
        excel_path, index=0, header=["Frame", "Agent", "Model", "Error", "Err_perc"]
    )

    figure_1, axis_1 = plt.subplots(2, 1)
    axis_1[0].plot(
        np.arange(0, len(model_angles)), model_angles, label="model_angles", color="red"
    )
    axis_1[1].plot(
        np.arange(0, len(agent_angles)),
        agent_angles,
        label="agent_angles",
        color="black",
    )
    axis_1[0].set_title("model_angles")
    axis_1[1].set_title("agent_angles")
    figure_1.set_size_inches(40.5, 15.5)
    plt.savefig(os.path.join(pkg_dir, "auto_pilotVsnvidia.png"))


# # ==============================================================================
# # -- Model Deployment ----------------------------------------------------------
# # ==============================================================================


# initiate class for preprocessing the image received from the camera
class PreprocessImage(object):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: transforms.functional.to_pil_image(img)
                    if not isinstance(img, Image.Image)
                    else img
                ),
                transforms.Lambda(
                    lambda img: transforms.functional.crop(
                        img, top=220, left=0, height=260, width=640
                    )
                ),  # Adjust as needed
                transforms.Resize((66, 200)),  # Adjust as needed
                transforms.ToTensor(),
                transforms.Lambda(lambda img: (img * 2.0) - 1.0),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


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

# initiate class for preprocessing the image received from the camera
preprocess_image = PreprocessImage()


# predict the steering angle
def predict_steering_angle(image):
    image = preprocess_image(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    steering_angle = model(image)
    steering_angle = steering_angle.item()
    steering_angle = steering_angle * 180 / np.pi
    steering_angle = steering_angle / 540
    return steering_angle


# ==============================================================================
# -- Main ----------------------------------------------------------------------
# ==============================================================================


def main():
    env = None
    # parameters for the carla_gym environment
    params = {
        "max_time_episode": 1000,  # maximum timesteps per episode
        "obs_size": [480, 640],  # observation (image) size[height,width]
        "min_speed": 10,  # desired minimum eg vehicle speed (Km/Hr)
        "max_speed": 15,  # desired maximum eg vehicle speed (Km/Hr)
        "discrete": False,  # whether to use discrete control space
        "discrete_steer": [-0.2, 0.0, 0.2],  # discrete value of steering angles
        "continuous_steer_range": [-1, 1],  # continuous steering angle range
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
                        obs["camera"]
                    )  # random action selection
                    print("Steering Angle: {:.3f}".format(model_steer))

                    action = [model_steer]
                    obs, reward, done, _, info = env.step(action)

                    score += reward

                else:
                    # Model Commands
                    model_steer = predict_steering_angle(
                        obs["camera"]
                    )  # random action selection
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
        generate_model_evaluation(frames, agent_angles, model_angles)

        if env != None:
            env.destroy_actors()
        print("\n>>> Cancelled by user. Bye!\n")

    finally:
        generate_model_evaluation(frames, agent_angles, model_angles)
        if env != None:
            env.destroy_actors()
        print("\n>>> Cancelled by user. Bye!\n")


if __name__ == "__main__":
    main()
