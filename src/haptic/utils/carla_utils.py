import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_model_evaluation(frames, agent_angles, model_angles, pkg_dir):
    """
    Create CSV file of dataset labels (Steering Angles in Deg)"""
    frame_lst = []
    agent_lst = []
    model_lst = []
    err_lst = []
    acc_lst = []
    for i in range(frames):
        frame_lst.append(str(i + 1))  # Form Frames Column
        agent_lst.append("%.5f" % agent_angles[i])  # Format float to 5 decimals
        model_lst.append("%.5f" % model_angles[i])  # Format float to 5 decimals
        err = abs(agent_angles[i] - model_angles[i])  # Form Error Column
        err_lst.append("%.5f" % err)  # Format float to 5 decimals
        acc = round(abs(err * 100 / agent_angles[i]))
        acc_lst.append(acc)

    # Convert lists to numpy arrays
    frame_lst = np.array(frame_lst)
    agent_lst = np.array(agent_lst)
    model_lst = np.array(model_lst)
    err_lst = np.array(err_lst)
    acc_lst = np.array(acc_lst)

    # Concatenate numpy arrays for CSV data
    data = np.stack((frame_lst, agent_lst, model_lst, err_lst, acc_lst), axis=1)

    # Write data to CSV file
    csv_path = os.path.join(pkg_dir, "results.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    # Convert CSV file to excel
    excel_path = os.path.join(pkg_dir, "results.xlsx")
    csv_file = pd.read_csv(csv_path)
    csv_file.to_excel(
        excel_path, index=0, header=["Frame", "Agent", "Model", "Error", "Err_perc"]
    )

    figure_1, axis_1 = plt.subplots(2, 1)
    axis_1[0].plot(
        np.arange(0, len(model_angles)), model_angles, label="model_angles", color="red"
    )
    axis_1[1].plot(
        np.arange(0, len(agent_angles)),
        agent_angles,
        label="agent_angles",
        color="black",
    )
    axis_1[0].set_title("model_angles")
    axis_1[1].set_title("agent_angles")
    figure_1.set_size_inches(40.5, 15.5)
    plt.savefig(os.path.join(pkg_dir, "auto_pilotVsnvidia.png"))


# initiate class for preprocessing the image received from the camera
# preprocess_image = PreprocessImage()


# predict the steering angle
def predict_steering_angle(image, model, device):
    # image = preprocess_image(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    steering_angle = model(image)
    steering_angle = steering_angle.item()/(540*np.pi/180)

    return steering_angle


def get_reward_comp(vehicle, waypoint, collision):
    vehicle_location = vehicle.get_location()
    x_wp = waypoint.transform.location.x
    y_wp = waypoint.transform.location.y

    x_vh = vehicle_location.x
    y_vh = vehicle_location.y

    wp_array = np.array([x_wp, y_wp])
    vh_array = np.array([x_vh, y_vh])

    dist = np.linalg.norm(wp_array - vh_array)

    vh_yaw = correct_yaw(vehicle.get_transform().rotation.yaw)
    wp_yaw = correct_yaw(waypoint.transform.rotation.yaw)
    cos_yaw_diff = np.cos((vh_yaw - wp_yaw)*np.pi/180.)

    collision = 0 if collision is None else 1
    
    return cos_yaw_diff, dist, collision

def reward_value(cos_yaw_diff, dist, collision, lambda_1=1, lambda_2=1, lambda_3=5):
    reward = (lambda_1 * cos_yaw_diff) - (lambda_2 * dist) - (lambda_3 * collision)
    return reward

def correct_yaw(x):
    return(((x%360) + 360) % 360)