import os
from time import time
from datetime import date
import os
import torch
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import PPO
from haptic.callbacks.carla_callbacks import (
    SaveBestModelCallback,
    PeriodicSaveModelCallback,
)
from haptic.architectures.custom_actor_critic import (
    CustomActorCriticPolicy,
    CNNFeatureExtractor,
)
from haptic.carla_gym.carla_env import CarlaEnv
from haptic.architectures.nvidia_arch import NetworkNvidia

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set the path for the model
# model_path = "/home/mtr-pbl/haptic/data_sets/best_model_1152_carla_18.pth"
model_path = "/home/mtr-pbl/haptic/data_sets/1152/best_model_1152_p_10.pth"

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))


if __name__ == "__main__":
    BUFFER_SIZE = 60_000
    TOTAL_TIMESTEPS = 1000_000
    EVAL_FREQ = 50_000  # number of timesteps between each evaluation
    RENDER_EVAL = False  # True if you want to render the evaluation
    OUTPUT_PATH = "./training_folder"  # path to save the training folder
    MODEL_SAVE_FREQ = 10_000  # number of timesteps between each model save

    today = date.today()
    date_str = today.strftime("%b-%d-%Y")
    train_folder_output_path = f"{OUTPUT_PATH}/{date_str}_{time()}"
    os.makedirs(train_folder_output_path)

    params = {
        "max_time_episode": 1000,  # maximum timesteps per episode
        "obs_size": [66, 200],  # observation (image) size[height,width]
        "min_speed": 10,  # desired minimum eg vehicle speed (Km/Hr)
        "max_speed": 15,  # desired maximum eg vehicle speed (Km/Hr)
        "discrete": False,  # whether to use discrete control space
        "discrete_steer": [-0.2, 0.0, 0.2],  # discrete value of steering angles
        "continuous_steer_range": [-1, 1],  # continuous steering angle range
        "cam_size": [480, 640],
        "scenario": "train",
        'dt': 0.1,  # time interval between two frames
    }
    env = CarlaEnv(params=params)

    policy_kwargs = dict(
        features_extractor_class=CNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=1152),
    )

    model = PPO(
        CustomActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=None,
        verbose=1,
        seed=None,
        device="auto",
        _init_setup_model=True,
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
        verbose=1
    )
    callbacks = CallbackList([save_best_model_callback, save_model_callback])

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    model.save(f"{train_folder_output_path}/ppo_carla")
