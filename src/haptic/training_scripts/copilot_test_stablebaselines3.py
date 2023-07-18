import matplotlib.pyplot as plt
from stable_baselines3.dqn.dqn import DQN
from haptic.gym.envs.box2d.car_racing import CarRacingSharedStablebaselines3, CarRacing
from haptic.gym import wrappers
from time import time

pilot_list = ["none", "laggy", "noisy", "optimal"]
avg_reward_dict = {"none": 0, "laggy": 0, "noisy": 0, "optimal": 0, "solo_pilot":0}
NO_EPISODES = 5
MAX_EPISODE_TIMESTEPS = 500

if __name__ == "__main__":
    t1 = time()
    for pilot in pilot_list:
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
            random_action_prob=0.2,
            laggy_pilot_freq=4,
        )
        env = wrappers.Monitor(env, f"./copilot_{pilot}_pilot_video/", force=True)
        model = DQN.load("copilot_stablebaselines3")
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
        # print("Total Timesteps =", total_timesteps)
        # print("Average Reward =", avg_reward)
        env.close()

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
    env = wrappers.Monitor(env, f"./solo_pilot_video/", force=True)
    model = DQN.load("trials/models/FINAL_MODEL_SMOOTH_STEERING_CAR")
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
    avg_reward_dict["solo_pilot"] = avg_reward
    # print("Total Timesteps =", total_timesteps)
    # print("Average Reward =", avg_reward)
    env.close()
    t2 = time()
    delta_t = (t2 - t1)/60
    print(f"took {delta_t} minutes")
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
    plt.savefig("Bar Diagram of Pilots Average Rewards")
