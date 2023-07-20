from stable_baselines3.dqn_copilot import CnnPolicyCopilot
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3 import DQN
from stable_baselines3 import DQNCopilot
from haptic.gym.envs.box2d.car_racing import CarRacingSharedStablebaselines3, CarRacing


if __name__ == "__main__":
    env = CarRacing(
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
    model = DQN(CnnPolicy, env=env, buffer_size=5000)
    model.learn(total_timesteps=100, log_interval=4)
    model.save("dqn_pilot")

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
        pilot="dqn_pilot",
        pilot_type=f"optimal_pilot",
        random_action_prob=0.2,
        laggy_pilot_freq=4,
    )

    model = DQNCopilot(CnnPolicyCopilot, env=env, buffer_size=5000)
    model.learn(total_timesteps=100, log_interval=4)
    model.save("dqn_copilot")
