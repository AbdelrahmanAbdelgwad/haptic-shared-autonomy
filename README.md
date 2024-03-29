![tests](https://github.com/AbdelrahmanAbdelgwad/haptic/workflows/test/badge.svg)
# Haptic
Shared autonomy using haptic feedback as a steering driver assistance system

In this repo, we are using the gym environment developed by NotAnyMike. You can find it on this link:
https://notanymike.github.io/Solving-CarRacing/

For the ROS implementation of the testing code, please visit this repository: 
https://github.com/AbdelrahmanAbdelgwad/haptic-shared-autonomy-ros-pkg


## Setup and Installation

`git clone git@github.com:AbdelrahmanAbdelgwad/haptic.git`

### Create a venv
Use the steps in this link:
https://code.visualstudio.com/docs/python/environments

### Rquirements

`cd haptic`

`pip install -r requirements.txt` 

`pip install -e .`

`git submodule update --init --recursive`

`git submodule update --remote`

`cd stable_baselines3`

`pip install -e .`

`cd ..`

`cd gym`

`pip install -e .`

`cd ..`

`cd src/haptic`


### Using the package
#### Training:

In the terminal, run the following command customized to your preference:

`haptic <mode> <policy_type> <path/to/pilot_model> <pilot_type> <copilot_model_name>`

- mode = "train"  
- policy_type = "Multi" or "Cnn"  
- pilot_type = "none_pilot","laggy_pilot", "noisy_pilot", "optimal_pilot", or "human_keyboard"

Example:

`haptic train Cnn FINAL_MODEL_SMOOTH_STEERING_CAR none_pilot copilot_2M_0.6_none_x0.3_x4_Cnn`

#### Testing:

`haptic <mode> <policy_type> <path/to/pilot_model> </path/to/copilot_model>`

- mode = "test"
- policy_type = "Multi" or "Cnn"

Example:

`haptic test Cnn FINAL_MODEL_SMOOTH_STEERING_CAR Copilot_600k_noisy_fine_tuning`


<!-- `pip install swig`

`$ git clone https://github.com/pybox2d/pybox2d pybox2d_dev`

`$ cd pybox2d_dev`

`$ python setup.py build`

`$ sudo python setup.py install`

`pip install box2d-py`

Replace .venv with the name of your environment and follow these steps:

"haptic-shared-autonomy/.venv/lib/python3.10/site-packages/stable_baselines3/common/vec_env/patch_gym.py", line 8
Replace the line with this :
    `import haptic.gym as gym  # pytype: disable=import-error`

"haptic-shared-autonomy/.venv/lib/python3.10/site-packages/shimmy/openai_gym_compatibility.py", line 35
Replace the line with this :
    `import haptic.gym as gym`

"haptic-shared-autonomy/.venv/lib/python3.10/site-packages/shimmy/openai_gym_compatibility.py", line 36
Replace the line with this :
    `import haptic.gym.wrappers`
Also replace all `gym.wrappers` with `haptic.gym.wrappers` -->




 

