import rospy
from std_msgs.msg import Float32, Int32
from haptic.gym import wrappers
from haptic.gym.envs.box2d.car_racing import CarRacingShared
from time import time
import torch as th
from haptic.learning_algorithm.shared_dqn_cnn import Agent
import numpy as np
from stable_baselines3 import DQN


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


def cont2disc(human_steering_action):
    action = 0
    if -0.05 < human_steering_action < 0.05:
        action = 0
    if -0.2 < human_steering_action < -0.05:
        action = 1  # LEFT_LEVEL_1
    if -0.4 < human_steering_action < -0.2:
        action = 2  # LEFT_LEVEL_2
    if -0.6 < human_steering_action < -0.4:
        action = 3  # LEFT_LEVEL_3
    if -0.8 < human_steering_action < -0.6:
        action = 4  # LEFT_LEVEL_4
    if -1 < human_steering_action < -0.8:
        action = 5  # LEFT_LEVEL_5
    if 0.05 < human_steering_action < 0.2:
        action = 6  # RIGHT_LEVEL_1
    if 0.2 < human_steering_action < 0.4:
        action = 7  # RIGHT_LEVEL_2
    if 0.4 < human_steering_action < 0.6:
        action = 8  # RIGHT_LEVEL_3
    if 0.6 < human_steering_action < 0.8:
        action = 9  # RIGHT_LEVEL_4
    if 0.8 < human_steering_action < 1:
        action = 10  # RIGHT_LEVEL_5
    return action


def main(alpha: float, total_timesteps: int, trial: str):
    score = 0
    n_actions = 15
    frames_per_state = 4
    STATE_W = 96
    STATE_H = 96
    model = th.load(
        "final_model_DQN_Car_Racer_alpha_0.6", map_location=th.device("cuda:0")
    )
    auto_model = DQN.load("FINAL_MODEL_SMOOTH_CAR")

    env = CarRacingShared(
        allow_reverse=False,
        grayscale=1,
        show_info_panel=1,
        discretize_actions="smooth",  # n_actions = 5
        num_tracks=2,
        num_lanes=2,
        num_lanes_changes=4,
        max_time_out=5,
        frames_per_state=frames_per_state,
    )
    agent = Agent(
        gamma=0.99,
        epsilon=0,
        batch_size=64,
        n_actions=n_actions,
        eps_end=0,
        input_dims=(96, 96, frames_per_state + 1),
        lr=0.003,
        max_mem_size=5000,
        max_q_target_iter=300,
        alpha=alpha,
        observation_space=env.observation_space,
        cuda_index=0,
    )
    agent.Q_pred = model
    agent.Q_pred.device = th.device(f"cuda:{0}" if th.cuda.is_available() else "cpu")
    # env = wrappers.Monitor(env, f"./video_{trial}/", force=True)
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
        # print(human_steering_action)
        state = (
            th.tensor(observation[:, :, 0:4]).to(agent.Q_pred.device).cpu().data.numpy()
        )
        action, _ = auto_model.predict(state)
        if action == 11 or action == 12 or action == 13 or action == 14:
            pi_action = action
        else:
            pi_action = cont2disc(human_steering_action)
        print(pi_action)
        env.render()
        pi_frame = pi_action * np.ones((STATE_W, STATE_H))
        observation[:, :, 4] = pi_frame
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action=action, pi_action=pi_action)
        score += reward
        observation = observation_
        # Publish agent_steering_action to "/agent" topic
        # agent_steering_pub.publish(agent_steering_action)
        score += reward
        score_pub.publish(score)
        # print(f"\n score of {total_timesteps} timessteps is {score} \n")
    # env.reset()
    env.close()


if __name__ == "__main__":
    alpha = 0.6
    total_timesteps = 1000
    trial_name = f"trial_10_alpha_{alpha}"
    main(alpha, total_timesteps, trial_name)
