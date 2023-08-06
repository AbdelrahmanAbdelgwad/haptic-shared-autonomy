import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.dqn_copilot import CnnPolicyCopilot
from stable_baselines3 import DQN
from stable_baselines3 import DQNCopilot
from haptic.gym.envs.box2d.car_racing import CarRacingSharedStablebaselines3, CarRacing
from haptic.gym import wrappers
from time import time

ALPHA = 1
RANDOM_ACTION_PROB = 0.2
LAG_FREQ = 4
copilot_pilot_list = [
    "none",
    "laggy",
    "noisy",
    "optimal",
]
pilot_list = [
    "solo_laggy",
    "solo_noisy",
    "solo_optimal",
]
avg_reward_dict = {
    "none": 0,
    "laggy": 0,
    "noisy": 0,
    "optimal": 0,
    "solo_laggy": 0,
    "solo_noisy": 0,
    "solo_optimal": 0,
}
NO_EPISODES = 5
MAX_EPISODE_TIMESTEPS = 1000

if __name__ == "__main__":
    t1 = time()
    result_data = []
    for pilot in pilot_list:
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
        env = wrappers.Monitor(
            env, f"./alpha_{ALPHA}/videos/{pilot}_pilot_video/", force=True
        )
        model = DQN.load("trials/models/FINAL_MODEL_SMOOTH_STEERING_CAR")
        # model = DQN(CnnPolicy, env=env, buffer_size=5000)
        # model.learn(total_timesteps=100, log_interval=4)
        # model.save("dqn_pilot")
        # del model  # remove to demonstrate saving and loading
        # model = DQN.load("dqn_pilot")
        episode_timesteps = 0
        done = False
        episode_reward = 0
        avg_reward = 0
        total_timesteps = 0
        laggy_pilot_counter = 0
        observation = env.reset()
        for episode in range(NO_EPISODES):
            while not done:
                episode_timesteps += 1
                total_timesteps += 1
                env.render()
                if pilot == "solo_noisy":
                    action, _ = model.predict(observation)
                    if np.random.random() < RANDOM_ACTION_PROB:
                        action = env.action_space.sample()
                elif pilot == "solo_laggy":
                    if laggy_pilot_counter % LAG_FREQ == 0:
                        action, _ = model.predict(observation)
                    laggy_pilot_counter += 1
                elif pilot == "solo_optimal":
                    action, _ = model.predict(observation)
                observation, reward, done, info = env.step(action)
                episode_reward += reward
                if done:
                    env.reset()
                    done = False
                if episode_timesteps % MAX_EPISODE_TIMESTEPS == 0:
                    episode_timesteps = 0
                    break
            avg_reward += episode_reward
        print(avg_reward)
        avg_reward_dict[pilot] = avg_reward
        result_data.append({
            "Pilot": pilot,
            "Average Reward": avg_reward,
            "Total Timesteps": total_timesteps,
        })
        env.close()

    for pilot in copilot_pilot_list:
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
            pilot_type=f"{pilot}_pilot",
            random_action_prob=RANDOM_ACTION_PROB,
            laggy_pilot_freq=LAG_FREQ,
        )
        env = wrappers.Monitor(
            env, f"./alpha_{ALPHA}/videos/copilot_{pilot}_pilot_video/", force=True
        )
        model = DQNCopilot.load("copilot_stablebaselines3")
        # model = DQNCopilot(CnnPolicyCopilot, env=env, buffer_size=5000)
        # model.learn(total_timesteps=100, log_interval=4)
        # model.save("dqn_copilot")
        # del model  # remove to demonstrate saving and loading
        # model = DQNCopilot.load("dqn_copilot")
        episode_timesteps = 0
        done = False
        episode_reward = 0
        avg_reward = 0
        total_timesteps = 0
        observation = env.reset()
        for episode in range(NO_EPISODES):
            while not done:
                episode_timesteps += 1
                total_timesteps += 1
                env.render()
                action, _ = model.predict(observation)
                observation, reward, done, info = env.step(action)
                episode_reward += reward
                if done:
                    env.reset()
                    done = False
                if episode_timesteps % MAX_EPISODE_TIMESTEPS == 0:
                    episode_timesteps = 0
                    break
            avg_reward += episode_reward
        avg_reward_dict[f"{pilot}"] = avg_reward
        result_data.append({
            "Pilot": pilot,
            "Average Reward": avg_reward,
            "Total Timesteps": total_timesteps,
        })
        env.close()

    t2 = time()
    delta_t = (t2 - t1) / 60
    print(f"took {delta_t} minutes")

    # Create a pandas DataFrame
    df = pd.DataFrame(result_data)

    # Save the DataFrame to a CSV file
    df.to_csv(f"./alpha_{ALPHA}/results_alpha_{ALPHA}.csv", index=False)

    pilots = list(avg_reward_dict.keys())
    rewards = list(avg_reward_dict.values())

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(pilots, rewards, color="blue", width=0.4)

    plt.xlabel("Pilots")
    plt.ylabel(f"Average rewards per {NO_EPISODES} episodes")
    plt.title(
        f"Pilots average rewards in {NO_EPISODES} episodes and {MAX_EPISODE_TIMESTEPS} timesteps per episode"
    )
    # plt.show()
    # plt.close()
    plt.savefig(f"./alpha_{ALPHA}/Bar Diagram of Pilots Average Rewards Alpha 1")