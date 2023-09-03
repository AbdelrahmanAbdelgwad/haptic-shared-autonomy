import haptic.gym as gym
from haptic.gym.envs.box2d.car_racing import CarRacing
from stable_baselines3 import DQN
import serial
import time
import matplotlib.pyplot as plt
import keyboard


def check_for_r_press():
    return keyboard.is_pressed("r")


def disc2cont(action):
    if action == 0:
        action = [0, 0, 0.0]  # NOTHINGserial
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


def main(alpha, trial):
    num = 0
    diff = 0
    model = DQN.load("DQN_model_hard_actions_2")
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
    env = gym.wrappers.Monitor(env, f"./video_{trial}/", force=True)
    observation = env.reset()
    data = []
    time.sleep(1)
    while True:
        time.sleep(0.001)
        user_input = serial.Serial("/dev/ttyACM0", 19200)
        print("data = ", user_input.readline().decode())
        human_steering_action = num / 90

        env.render()
        action, _ = model.predict(observation)
        action = disc2cont(action)
        agent_steering_action = action[0]
        mixed_steering_action = (
            1 - alpha
        ) * agent_steering_action + alpha * human_steering_action
        action[0] = mixed_steering_action
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
        line = user_input.readline()  # read a byte string
        if line:
            string = line.decode()  # convert the byte string to a unicode string
            num = int(string)  # convert the unicode string to an int
            # print(string)
            data.append(num)  # add int to data list
        # if check_for_r_press:
        #     env.reset()
        # diff = agent_steering_action - human_steering_action
        # diff = bytearray(struct.pack("f", diff))
        # haptic_feedback.write(diff)

    user_input.close()

    # build the plot
    plt.plot(data)
    plt.xlabel("timesteps")
    plt.ylabel("IMU readings")
    plt.title("IMU readings vs. timesteps")
    # plt.show()
    plt.savefig(f"graph_of_trial{trial}.png")
    # plt.close()


if __name__ == "__main__":
    alpha = 0.5
    trial = 7
    # for trial in range(6, 7):
    print("\n======== this is trial ", trial, "========\n")
    print("alpha value =", alpha)
    # if alpha > 1:
    #     break
    main(alpha, trial)
    # alpha += 0.2
