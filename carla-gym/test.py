# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import gym
import carla_gym
import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import random_split



# ==============================================================================
# -- Global Parameters ---------------------------------------------------------
# ==============================================================================

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set the path for the model
model_path = "/home/mtr-pbl/haptic/data_sets/1152/best_model_1152_carla.pth"



# ==============================================================================
# -- Model Deployment ----------------------------------------------------------
# ==============================================================================

# initiate class for preprocessing the image received from the camera
class PreprocessImage(object):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: transforms.functional.to_pil_image(img) if not isinstance(img, Image.Image) else img),
            transforms.Lambda(lambda img: transforms.functional.crop(img, top=220, left=0, height=260, width=640)),  # Adjust as needed
            transforms.Resize((66, 200)),  # Adjust as needed
            transforms.ToTensor(),
            transforms.Lambda(lambda img: (img * 2.0) - 1.0)
        ])

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
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=1152, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1)
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
    image=image.to(device)
    steering_angle = model(image)
    steering_angle = steering_angle.item()
    steering_angle = steering_angle/ np.pi
    return steering_angle



# ==============================================================================
# -- Main ----------------------------------------------------------------------
# ==============================================================================

def main():
    env = None
    # parameters for the carla_gym environment
    params = {
        'max_time_episode': 1000,  # maximum timesteps per episode
        'obs_size': [480, 640], # observation (image) size[height,width]
        'min_speed': 10, # desired minimum eg vehicle speed (Km/Hr)
        'max_speed': 15, # desired maximum eg vehicle speed (Km/Hr)
        'discrete': False,  # whether to use discrete control space
        'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
        'continuous_steer_range': [-1, 1],  # continuous steering angle range
    }
  
    try:
        # Set carla-gym environment
        env = gym.make('Carla-v0', params=params)
        episodes = 10_000

        for i in range(1, episodes+1):
            
            obs, info = env.reset() # MOD: added info extraction
            if i < 2:
                cv2.imwrite("a.jpg", obs["camera"])
            done = False
            score = 0
            
            while not done:
                steering_angle = predict_steering_angle(obs["camera"]) # random action selection
                print("Steering Angle: {:.3f}".format(steering_angle))
                
                action = [steering_angle]        
                obs, reward, done, _, info = env.step(action)                                

                score += reward
                # print(f"Collision Histroy: {info['Collision']}")
                # print(f"Lane INvasion Timestamps: {info['LaneInv']}")
            
            # print(f"Final Score: {score}")


    except KeyboardInterrupt:
        if env != None:
            env.destroy_actors()
        print('\n>>> Cancelled by user. Bye!\n')
    

    finally:
        if env != None:
            env.destroy_actors()
        print('\n>>> Cancelled by user. Bye!\n')


if __name__ == '__main__':

    main()