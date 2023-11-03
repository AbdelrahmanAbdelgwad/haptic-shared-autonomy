import rospy
from typing import List
from std_msgs.msg import Float32, Int16
from haptic.gym import wrappers
from haptic.gym.envs.box2d.car_racing import CarRacing
from time import time
from haptic.learning_algorithm.shared_dqn_cnn import Agent
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN


def disc2cont(action):
    if action == 0:
        action = [0, 0.2, 0.05]  # "NOTHING"
    if action == 1:
        action = [-0.2, 0.2, 0.05]  # LEFT_LEVEL_1
    if action == 2:
        action = [-0.4, 0.2, 0.05]  # LEFT_LEVEL_2
    if action == 3:
        action = [-0.6, 0.2, 0.05]  # LEFT_LEVEL_3
    if action == 4:
        action = [-0.8, 0.2, 0.05]  # LEFT_LEVEL_4
    if action == 5:
        action = [-1, 0.2, 0.05]  # LEFT_LEVEL_5
    if action == 6:
        action = [0.2, 0.2, 0.05]  # RIGHT_LEVEL_1
    if action == 7:
        action = [0.4, 0.2, 0.05]  # RIGHT_LEVEL_2
    if action == 8:
        action = [0.6, 0.2, 0.05]  # RIGHT_LEVEL_3
    if action == 9:
        action = [0.8, 0.2, 0.05]  # RIGHT_LEVEL_4
    if action == 10:
        action = [1, 0.2, 0.05]  # RIGHT_LEVEL_5
    return action


def cont2disc(human_steering_action):
    action = 0
    if -0.05 <= human_steering_action <= 0.05:
        action = 0
    if -0.2 <= human_steering_action < -0.05:
        action = 1  # LEFT_LEVEL_1
    if -0.4 <= human_steering_action < -0.2:
        action = 2  # LEFT_LEVEL_2
    if -0.6 <= human_steering_action < -0.4:
        action = 3  # LEFT_LEVEL_3
    if -0.8 <= human_steering_action < -0.6:
        action = 4  # LEFT_LEVEL_4
    if -1 <= human_steering_action < -0.8:
        action = 5  # LEFT_LEVEL_5
    if 0.05 < human_steering_action <= 0.2:
        action = 6  # RIGHT_LEVEL_1
    if 0.2 < human_steering_action <= 0.4:
        action = 7  # RIGHT_LEVEL_2
    if 0.4 < human_steering_action <= 0.6:
        action = 8  # RIGHT_LEVEL_3
    if 0.6 < human_steering_action <= 0.8:
        action = 9  # RIGHT_LEVEL_4
    if 0.8 < human_steering_action <= 1:
        action = 10  # RIGHT_LEVEL_5
    return action


def get_feedback(pi_action, opt_action):
    pi_steering = disc2cont(pi_action)[0]
    opt_steering = disc2cont(opt_action)[0]
    if (
        ((pi_steering < 0) and (opt_steering < 0))
        or ((pi_steering > 0) and (opt_steering > 0))
        or ((pi_steering == 0) and (opt_steering == 0))
    ):
        return 0
    elif (pi_steering >= 0) and (opt_steering < 0):
        return -1
    elif (pi_steering <= 0) and (opt_steering > 0):
        return 1
    elif (pi_steering > 0) and (opt_steering == 0):
        return -1
    elif (pi_steering < 0) and (opt_steering == 0):
        return 1
    else:
        return 0


def main(
    alpha_schedule: List,
    total_timesteps: int,
    methods_schedule: List,
    feedback: bool,
    user_name: str,
    trial: int,
):
    results = {}
    results_df = pd.DataFrame(columns=["Method", "Alpha", "Score"])
    frames_per_state = 4
    model = DQN.load("FINAL_MODEL_SMOOTH_STEERING_CAR_VALEO")
    abs_timestep = 0
    if feedback:
        feedback_str = "feedback"
    else:
        feedback_str = "no_feedback"
    for method in methods_schedule:
        for alpha in alpha_schedule:
            score = 0
            env = CarRacing(
                allow_reverse=False,
                grayscale=1,
                show_info_panel=1,
                discretize_actions="smooth_steering",  # n_actions = 11
                num_tracks=1,
                num_lanes=2,
                num_lanes_changes=4,
                max_time_out=5,
                frames_per_state=frames_per_state,
                display=f"Method {method} - Alpha = {alpha}",
            )

            env = wrappers.Monitor(
                env,
                f"./data_collected_{trial}/{user_name}/{feedback_str}/{method}/video/alpha_{alpha}/",
                force=True,
            )
            observation = env.reset()

            rospy.init_node("car_control_node")
            feedback_pub = rospy.Publisher("/feedback", Int16, queue_size=10)
            score_pub = rospy.Publisher("/score", Float32, queue_size=10)
            alpha_pub = rospy.Publisher("/alpha", Float32, queue_size=10)
            done = False
            timestep = 0
            human_counter = 0
            agent_counter = 0
            human_action = []
            while not done:
                msg = rospy.wait_for_message("/counter", Int16)
                if abs_timestep == 0:
                    zero_counter = msg.data
                human_steering_action = min((msg.data - zero_counter) / 1100, 1)
                human_steering_action = max(human_steering_action, -1)
                pi_action = cont2disc(human_steering_action)
                opt_action, _ = model.predict(observation)
                human_action.append(human_steering_action)
                if method == "RL":
                    # Assuming 'observation' is a numpy array with shape (96, 96, 4)
                    # Transpose the array to match the expected input format (batch_size, channels, height, width)
                    observation = observation.transpose((2, 0, 1))
                    # Convert the numpy array to a PyTorch tensor
                    observation_tensor = th.tensor(observation, dtype=th.float32).to(
                        "cuda"
                    )
                    # Add batch dimension
                    observation_tensor = observation_tensor.unsqueeze(0)
                    # Pass the observation tensor to the model
                    q_values = model.policy.q_net.forward(observation_tensor)
                    q_values -= th.min(q_values)

                    pi_action_q_value = q_values[0][pi_action]
                    opt_action_q_value = q_values[0][opt_action]

                    if pi_action_q_value >= (1 - alpha) * opt_action_q_value:
                        action = pi_action
                        print("human")
                        human_counter += 1
                    else:
                        action = opt_action
                        print("agent")
                        agent_counter += 1
                    action = disc2cont(action)
                    action[0] = human_steering_action
                elif method == "PIM":
                    agent_steering_action = disc2cont(opt_action)[0]
                    action_steering = (
                        alpha * human_steering_action
                        + (1 - alpha) * agent_steering_action
                    )
                    action = disc2cont(opt_action)
                    action[0] = action_steering
                
                observation_, reward, done, info = env.step(action)
                env.render()
                score += reward
                observation = observation_
                
                # if abs(disc2cont(opt_action)[0] - human_steering_action) >= 0.2:
                #     alpha-=0.008    
                # else:
                #     alpha+=0.008
                # if alpha >= 1:
                #     alpha =1
                # elif alpha <=0:
                #     alpha = 0
                # alpha_pub.publish(alpha)
                if feedback:
                    feedback_value = get_feedback(pi_action, opt_action)
                    feedback_pub.publish(feedback_value)

                score_pub.publish(score)

                print("timestep is", timestep, "\n")
                if done and (timestep < total_timesteps):
                    env.reset()
                    done = False

                if timestep >= total_timesteps:
                    if feedback:
                        feedback_value = 0
                        feedback_pub.publish(feedback_value)
                    break

                if rospy.is_shutdown():
                    if feedback:
                        feedback_value = 0  
                        feedback_pub.publish(feedback_value)

                timestep += 1
                abs_timestep += 1
            # print("human percent", human_counter / total_timesteps)
            # print("agent percent", agent_counter / total_timesteps)
                

                
            env.close()
            results[f"{method}_alpha_{alpha}"] = score
            results_df = results_df.append(
                {"Method": method, "Alpha": alpha, "Score": score}, ignore_index=True
            )

            #create human_action plot
            plt.figure(figsize=(10, 5))

            plt.plot(human_action)
            plt.xlabel("timesteps")
            plt.ylabel("Human Actions")
            plt.title("Human Actions vs. timesteps")
            # plt.show()
            plt.savefig(f"./data_collected_{trial}/{user_name}/{feedback_str}/{method}/human_actions_of_alpha_{alpha}.jpeg")

        pilots = list(results.keys())
        rewards = list(results.values())
        results = {}

        fig = plt.figure(figsize=(10, 5))

        # creating the bar plot
        plt.bar(pilots, rewards, color="blue", width=0.4)


        plt.xlabel("methods")
        plt.ylabel(f"score per {total_timesteps} timesteps")
        plt.title(f"various setups scores in {total_timesteps} timesteps")
        # plt.show()
        # plt.close()
        plt.savefig(
            f"./data_collected_{trial}/{user_name}/{feedback_str}/{method}/bar_chart_for_scores_of_alpha_{alpha}"
        )


    results_df.to_csv(
        f"./data_collected_{trial}/{user_name}/{feedback_str}/results.csv", index=False
    )

    
if __name__ == "__main__":
    # methods_schedule = ["PIM", "RL"]
    methods_schedule = ["RL"]
    
    # alpha_schedule = [0, 0.3, 0.6, 1]
    alpha_schedule = [0.7]
    total_timesteps = 10000
    # total_timesteps = 1000

    feedback = True
    user_name = "hydra"
    trial = 1
    main(alpha_schedule, total_timesteps, methods_schedule, feedback, user_name, trial)
