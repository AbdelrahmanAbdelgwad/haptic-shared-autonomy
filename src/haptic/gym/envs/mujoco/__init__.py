from haptic.gym.envs.mujoco.mujoco_env import MujocoEnv

# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from haptic.gym.envs.mujoco.ant import AntEnv
from haptic.gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from haptic.gym.envs.mujoco.hopper import HopperEnv
from haptic.gym.envs.mujoco.walker2d import Walker2dEnv
from haptic.gym.envs.mujoco.humanoid import HumanoidEnv
from haptic.gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from haptic.gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from haptic.gym.envs.mujoco.reacher import ReacherEnv
from haptic.gym.envs.mujoco.swimmer import SwimmerEnv
from haptic.gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from haptic.gym.envs.mujoco.pusher import PusherEnv
from haptic.gym.envs.mujoco.thrower import ThrowerEnv
from haptic.gym.envs.mujoco.striker import StrikerEnv
