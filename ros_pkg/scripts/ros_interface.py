import rospy
from std_msgs.msg import Float32, Int32
import haptic.gym as gym
from haptic.gym.envs.box2d.car_racing import CarRacing
from stable_baselines3 import DQN
from time import time


def disc2cont(action):
    if action == 0:
        action = [0, 0, 0.0]  # "NOTHING"
    if action == 1:
        action = [-0.2, 0, 0.0]  # LEFT_LEVEL_1
    if action == 2:
        action = [-0.4, 0, 0.0]  # LEFT_LEVEL_2
    if action == 3:
        action = [-0.6, 0, 0.0]  # LEFT_LEVEL_3
    if action == 4:
        action = [-0.8, 0, 0.0]  # LEFT_LEVEL_4
    if action == 5:
        action = [-1, 0, 0.0]  # LEFT_LEVEL_5
    if action == 6:
        action = [0.2, 0, 0.0]  # RIGHT_LEVEL_1
    if action == 7:
        action = [0.4, 0, 0.0]  # RIGHT_LEVEL_2
    if action == 8:
        action = [0.6, 0, 0.0]  # RIGHT_LEVEL_3
    if action == 9:
        action = [0.8, 0, 0.0]  # RIGHT_LEVEL_4
    if action == 10:
        action = [1, 0, 0.0]  # RIGHT_LEVEL_5
    if action == 11:
        action = [0, +0.5, 0.0]  # SOFT_ACCELERATE
    if action == 12:
        action = [0, +1, 0.0]  # HARD_ACCELERATE
    if action == 13:
        action = [0, 0, 0.4]  # SOFT_BREAK
    if action == 14:
        action = [0, 0, 0.8]  # HARD_BREAK

    return action


def main(alpha: float, total_timesteps: int, trial: str):
    score = 0
    model = DQN.load("FINAL_MODEL_SMOOTH_CAR")
    env = CarRacing(
        allow_reverse=False,
        grayscale=1,
        show_info_panel=1,
        discretize_actions="smooth",
        num_tracks=2,
        num_lanes=2,
        num_lanes_changes=4,
        max_time_out=0,
        frames_per_state=4,
    )
    env = gym.wrappers.Monitor(env, f"./video_{trial}/", force=True)
    observation = env.reset()

    rospy.init_node("car_control_node")
    agent_steering_pub = rospy.Publisher("/agent", Float32, queue_size=10)
    score_pub = rospy.Publisher("/score", Float32, queue_size=10)

    # rospy.spin()
    # while not rospy.is_shutdown():
    for timestep in range(total_timesteps):
        msg = rospy.wait_for_message("/counter", Int32)
        if timestep == 0:
            zero_counter = msg.data
        # rospy.loginfo(msg)
        human_steering_action = min(-(msg.data - zero_counter) / 600, 1)
        human_steering_action = max(human_steering_action, -1)
        print(human_steering_action)
        env.render()
        action, _ = model.predict(observation)
        action = disc2cont(action)
        agent_steering_action = action[0]
        mixed_steering_action = (
            1 - alpha
        ) * agent_steering_action + alpha * human_steering_action
        action[0] = mixed_steering_action
        observation, reward, done, info = env.step(action)
        # Publish agent_steering_action to "/agent" topic
        agent_steering_pub.publish(agent_steering_action)
        score += reward
        score_pub.publish(score)
        # print(f"\n score of {total_timesteps} timessteps is {score} \n")
    # env.reset()
    env.close()


if __name__ == "__main__":
    alpha = 0.6
    total_timesteps = 1000
    trial_name = f"trial_9_alpha_{alpha}"
    main(alpha, total_timesteps, trial_name)
