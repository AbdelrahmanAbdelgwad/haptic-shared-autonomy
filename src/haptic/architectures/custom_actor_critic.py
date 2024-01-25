from time import time
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import os
import gym
from gym.envs.box2d.car_racing import CarRacing
from gym import wrappers
import torch as th
from torch import nn
import torch.nn.functional as F
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# from stable_baselines3.ddpg.ddpg import DDPG
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from haptic.callbacks.car_racing_callbacks import (
    SaveBestModelCallback,
    PeriodicSaveModelCallback,
)
from haptic.architectures.nvidia_arch import NetworkNvidia


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 50):
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim)

        nvidia_net = NetworkNvidia()
        self.cnn = nvidia_net.conv_layers

    def forward(self, observations: th.Tensor) -> th.Tensor:
        output = self.cnn(observations)
        flatten_output = th.flatten(output, start_dim=1)
        return flatten_output


class CustomNetworkActorCritic(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int = 1152,
        last_layer_dim_pi: int = 1,
        last_layer_dim_vf: int = 1,
    ):
        super(CustomNetworkActorCritic, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        nvidia_net = NetworkNvidia()

        # Policy network
        self.policy_net = nvidia_net.linear_layers

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(in_features=1152, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetworkActorCritic(self.features_dim)


if __name__ == "__main__":
    # TODO: Reward shaping as they did and max episode steps as well -> Read many many papers
    # TODO: Look Reward fucntions in Carla Gym Env Repo

    # TODO: Implement gradient clipping if needed
    # TODO: Make sure that the observation is normalized

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
    TOTAL_TIMESTEPS = 2000_000

    EVAL_FREQ = 100_000  # number of timesteps between each evaluation
    RENDER_EVAL = False  # True if you want to render the evaluation
    OUTPUT_PATH = "./training_folder"  # path to save the training folder
    MODEL_SAVE_FREQ = 100_000  # number of timesteps between each model save

    TRAIN = False

    NO_EPISODES = 1000
    BETA = 1

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

    if TRAIN:
        today = date.today()
        date_str = today.strftime("%b-%d-%Y")
        train_folder_output_path = f"{OUTPUT_PATH}/{date_str}_{time()}"
        os.makedirs(train_folder_output_path)

        policy_kwargs = dict(
            features_extractor_class=CNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=1152),
        )

        model = PPO(
            CustomActorCriticPolicy,
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            learning_rate=0.0001,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
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

        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

        model.save("actor_critic_car_racing")

    else:
        env = wrappers.Monitor(
            env,
            f"./videos_ppo/",
            force=True,
        )
        model = PPO.load("training_folder/Jan-21-2024_1705789568.5404365/best_model")
        episode_timesteps = 0
        done = False
        episode_reward = 0
        total_timesteps = 0
        total_reward = 0
        observation = env.reset()
        action_prev = 0
        for episode in range(NO_EPISODES):
            while not done:
                episode_timesteps += 1
                total_timesteps += 1
                env.render()
                action, _ = model.predict(observation)
                action = BETA * action + (1 - BETA) * action_prev
                action_prev = action
                observation, reward, done, info = env.step(action)
                episode_reward += reward
                total_reward += reward
                if done:
                    env.reset()
                    done = False
        env.close()
    # action_noise = OrnsteinUhlenbeckActionNoise(
    #     mean=np.array([0]),
    #     sigma=np.array([0.4]),
    #     theta=np.array([0.6]),
    #     dt=1e-2,
    #     initial_noise=None,
    #     dtype=np.float32,
    # )
    # model = DDPG(
    #     "MlpPolicy",
    #     env,
    #     policy_kwargs=policy_kwargs,
    #     verbose=1,
    #     buffer_size=BUFFER_SIZE,
    #     batch_size=64,
    #     gamma=0.9,
    #     action_noise=action_noise,
    #     device="cuda",
    # )
