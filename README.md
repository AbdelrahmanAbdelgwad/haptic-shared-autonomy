![tests](https://github.com/AbdelrahmanAbdelgwad/haptic/workflows/test/badge.svg)
# Haptic
Haptic feedback controll to assistant driving systems


## Installation

create a venv using the steps in this link:
https://code.visualstudio.com/docs/python/environments

`pip install -r requirements.txt`

`pip install swig`

`$ git clone https://github.com/pybox2d/pybox2d pybox2d_dev`

`$ cd pybox2d_dev`

`$ python setup.py build`

`$ sudo python setup.py install`

`pip install box2d-py`

Replace .venv with the name of your environment and follow these steps:

"haptic-shared-autonomy/.venv/lib/python3.10/site-packages/stable_baselines3/common/vec_env/patch_gym.py", line 8
Replace the line with this :
    `import haptic.gym as gym  # pytype: disable=import-error`
 

