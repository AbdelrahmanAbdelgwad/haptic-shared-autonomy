import rospy
from std_msgs.msg import Float32
import haptic.gym as gym
from haptic.gym.envs.box2d.car_racing import CarRacing
from stable_baselines3 import DQN


def disc2cont(action):
    if action == 0:
        action = [0, 0, 0.0]  # NOTHING
    if action == 1:
        action = [-0.5, 0, 0.0]  # SOFT_LEFT
    if action == 2:
        action = [-1, 0, 0.0]  # HARD_LEFT
    if action == 3:
        action = [+0.5, 0, 0.0]  # SOFT_RIGHT
    if action == 4:
        action = [+1, 0, 0.0]  # HARD_RIGHT
    if action == 5:
        action = [0, +0.5, 0.0]  # SOFT_ACCELERATE
    if action == 6:
        action = [0, +1, 0.0]  # HARD_ACCELERATE
    if action == 7:
        action = [0, 0, 0.4]  # SOFT_BREAK
    if action == 8:
        action = [0, 0, 0.8]  # HARD_BREAK
    return action


env = CarRacing(
    allow_reverse=False,
    grayscale=1,
    show_info_panel=1,
    discretize_actions="soft",
    num_tracks=2,
    num_lanes=2,
    num_lanes_changes=4,
    max_time_out=0,
    frames_per_state=4,
)

# Uncomment following line to save video of our Agent interacting in this environment
# This can be used for debugging and studying how our agent is performing
env = gym.wrappers.Monitor(env, "./video/", force=True)
model = DQN.load("DQN_model_hard_actions_2")
t = 0
done = False
episode_reward = 0
observation = env.reset()

# Notice that episodes here are very small due to the way that the environment is structured
for episode in range(1):
    while not done:
        t += 1
        env.render()
        human_steering_action = 0.1
        action, _ = model.predict(observation)
        action = disc2cont(action)
        agent_steering_action = action[0]
        mixed_steering_action = (
            0.5 * agent_steering_action + 0.5 * human_steering_action
        )
        action[0] = mixed_steering_action
        observation, reward, done, info = env.step(action)
        episode_reward += reward

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

    print(episode_reward)
    episode_reward = 0

env.close()
