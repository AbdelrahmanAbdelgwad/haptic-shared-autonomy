from stable_baselines3.dqn_copilot import CnnPolicyCopilot
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3 import DQN
from stable_baselines3 import DQNCopilot
from haptic.gym import wrappers
from haptic.gym.envs.box2d.car_racing import CarRacingSharedStablebaselines3, CarRacing

RANDOM_ACTION_PROB = 1
LAG_FREQ = 4
TOTAL_TIMESTEPS = 4000_000

if __name__ == "__main__":
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
        pilot_type=f"noisy_pilot",
        random_action_prob=RANDOM_ACTION_PROB,
        laggy_pilot_freq=LAG_FREQ,
    )
    model = DQNCopilot(CnnPolicyCopilot, env=env, buffer_size=100000, verbose=1)
    # model = DQNCopilot.load("copilot_stablebaselines3", env =)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4)
    model.save("copilot_stablebaselines3")
