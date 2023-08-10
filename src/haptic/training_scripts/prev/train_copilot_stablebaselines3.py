from gym.envs.box2d.car_racing import CarRacingSharedStablebaselines3
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn import CnnPolicy

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
    pilot_type="laggy_pilot",
    random_action_prob=0.2,
    laggy_pilot_freq=4,
)

# model = DQN(CnnPolicy, env, buffer_size=100_000, verbose=1, copilot=True)
model = DQN.load("copilot_stablebaselines3", env=env)
model.learn(total_timesteps=1000_000, log_interval=4)
model.save("copilot_stablebaselines3")
