from stable_baselines3 import DQN
from stable_baselines3.dqn import CnnPolicy
from haptic.gym.envs.box2d.car_racing import CarRacingSharedStablebaselines3

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
)

model = DQN(CnnPolicy, env, buffer_size=5000, verbose=1, copilot=True)
model.learn(total_timesteps=500_000, log_interval=4)
model.save("copilot_stablebaselines3")
