import os
import csv
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


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


# # ==============================================================================
# # -- Model Deployment ----------------------------------------------------------
# # ==============================================================================


# initiate class for preprocessing the image received from the camera
class PreprocessImage(object):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: transforms.functional.to_pil_image(img)
                    if not isinstance(img, Image.Image)
                    else img
                ),
                transforms.Lambda(
                    lambda img: transforms.functional.crop(
                        img, top=220, left=0, height=260, width=640
                    )
                ),  # Adjust as needed
                transforms.Resize((66, 200)),  # Adjust as needed
                transforms.ToTensor(),
                transforms.Lambda(lambda img: (img * 2.0) - 1.0),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


# initiate class for preprocessing the image received from the camera
preprocess_image = PreprocessImage()


# predict the steering angle
def predict_steering_angle(image, model, device):
    image = preprocess_image(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    steering_angle = model(image)
    steering_angle = steering_angle.item()
    steering_angle = steering_angle * 180 / np.pi
    steering_angle = steering_angle / 540
    return steering_angle
