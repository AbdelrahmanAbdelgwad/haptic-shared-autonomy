# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import sys
import signal
import random
import carla
import cv2
import time
import math
import numpy as np
import pandas as pd
from gym import Env, spaces
from gym.utils import seeding
from skimage.transform import resize
from agents.navigation.basic_agent import BasicAgent
from torchvision import transforms
from PIL import Image


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
                # transforms.Lambda(lambda img: (img * 2.0) - 1.0),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


# ==============================================================================
# -- Carla-Env Class -----------------------------------------------------------
# ==============================================================================


class CarlaEnv(Env):
    steer_amp = 1.0
    show_preview = False

    def __init__(self, params):
        # Parameters
        self.throttle_val = 0.5
        self.max_time_episode = params["max_time_episode"]
        self.min_speed = params["min_speed"]
        self.max_speed = params["max_speed"]
        self.obs_size = [
            params["obs_size"][0],
            params["obs_size"][1],
        ]  # MOD: height, width
        self.cam_size = [params["cam_size"][0], params["cam_size"][1]]
        self.actor_filters = [
            "sensor.other.collision",
            "sensor.camera.rgb",
            "vehicle.*",
        ]

        # Action Space
        self.discrete = params["discrete"]
        self.discrete_act = [params["discrete_steer"]]  # steer only
        self.n_steer = len(self.discrete_act[0])
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_steer)
        else:
            self.action_space = spaces.Box(
                np.float32(np.array([params["continuous_steer_range"][0]])),
                np.float32(np.array([params["continuous_steer_range"][1]])),
                shape=(1,),
                dtype=np.float32,
            )

        # Observation Space
        # observation_space_dict = {
        # 'camera': spaces.Box(low=0, high=255, shape=(self.obs_size[0], self.obs_size[1], 3), dtype=np.uint8)
        # }
        # self.observation_space = spaces.Dict(observation_space_dict)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.obs_size[0], self.obs_size[1]),
            dtype=np.uint8,
        )

        # Connect to Carla server
        print("\nConneting To Simulator ...\n")
        signal.signal(signal.SIGINT, self._signal_handler)
        i = 1
        while True:
            try:
                self.client = carla.Client("localhost", 2000)
                self.client.set_timeout(5.0)
                self.world = self.client.load_world("Town07")
                print("\n" + "\u2713" * 3 + " Connected Successfully")
                break
            except RuntimeError:
                print("\u2715" + f" Connection failed, trying again: {i}")
                i += 1

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # Actors List
        self.actor_list = []

        # Ego Vehicle Blueprint
        self.blueprint_library = self.world.get_blueprint_library()
        self.model3 = self.blueprint_library.filter("model3")[0]

        # RGB Camera Sensor
        self.cam_img = np.zeros((self.cam_size[0], self.cam_size[1], 3), dtype=np.uint8)
        self.cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        self.cam_trans = carla.Transform(carla.Location(x=2.5, z=0.7))
        # Configure RGB Camera Attributes
        self.cam_bp.set_attribute("image_size_x", f"{self.cam_size[1]}")
        self.cam_bp.set_attribute("image_size_y", f"{self.cam_size[0]}")
        self.cam_bp.set_attribute("fov", "110")
        self.cam_bp.set_attribute("sensor_tick", "0.01")

        # Collision Sensor
        self.collision_hist = []
        self.collision_hist_l = 1  # collision histroy length
        self.collision_bp = self.blueprint_library.find("sensor.other.collision")
        self.col_trans = carla.Transform(carla.Location(x=2.5, z=0.7))

        # Lane Invasion
        self.laneInv_list = []
        self.laneInv_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.laneInv_trans = carla.Transform(carla.Location(x=2.5, z=0.7))

        # Set fixed simulation step for synchronous mode
        # self.settings = self.world.get_settings()
        # self.settings.fixed_delta_seconds = self.dt

        self.preprocess_image = PreprocessImage()
        self.reward = 0
        self.episode_steps = 0
        self.total_reward = 0
        self.episode_reward = 0
        self.scenario = params["scenario"]
        self.statistics = pd.DataFrame(
            columns=[
                "total_steps",
                "episode_steps",
                "scenario",
                "total_reward",
                "episode_reward",
                "reward",
            ]
        )

    def reset(self, seed=None, options=None):
        """This method intialize world for new epoch"""
        self.episode_steps = 0
        self.episode_reward = 0
        # Clear Sensor Objects
        self.camera_sensor = None
        self.collision_sensor = None
        self.laneInv_sensor = None

        # Clear Sensors History
        self.collision_hist = []
        self.laneInv_list = []

        # Delete sensors, vehicles and walkers
        self.destroy_actors()

        # Spawn Ego Vehicle
        ego_trans = random.choice(self.world.get_map().get_spawn_points())
        self.ego_vehicle = self.world.spawn_actor(self.model3, ego_trans)
        self.ego_vehicle.role_name = "ego_vehicle"
        self.actor_list.append(self.ego_vehicle)
        # Turn Ego-vehicle into Basic Agent
        self.agent = BasicAgent(self.ego_vehicle)
        self.agent.ignore_traffic_lights(active=True)
        time.sleep(3)

        # Spawn RGB Camera
        self.camera_sensor = self.world.spawn_actor(
            self.cam_bp, self.cam_trans, attach_to=self.ego_vehicle
        )
        self.actor_list.append(self.camera_sensor)
        self.camera_sensor.listen(lambda data: get_cam_img(data))

        # RGB Camera Callback
        def get_cam_img(data):
            """
            Process RGB camera data"""
            i = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            i2 = np.reshape(i, (data.height, data.width, 4))
            i3 = i2[:, :, :3]
            # i3 = i3[:, :, ::-1] # reverse sequence
            self.cam_img = i3
            if CarlaEnv.show_preview:
                cv2.imshow("", i3)
                cv2.waitKey(1)
            self.cam_img = i3

        # Spawn Collision Sensor
        self.col_sensor = self.world.spawn_actor(
            self.collision_bp, self.col_trans, attach_to=self.ego_vehicle
        )
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: get_col_hist(event))

        # Collision Sensor Callback
        def get_col_hist(event):
            """
            Add impulse to collision history"""
            impulseX = event.normal_impulse.x
            impulseY = event.normal_impulse.y
            impulseZ = event.normal_impulse.z
            intensity = math.sqrt((impulseX**2) + (impulseY**2) + (impulseZ**2))
            self.collision_hist.append(intensity)

        # Spawn Lane Invasion Detector
        self.laneInv_sensor = self.world.spawn_actor(
            self.laneInv_bp, self.laneInv_trans, attach_to=self.ego_vehicle
        )
        self.actor_list.append(self.laneInv_sensor)
        self.laneInv_sensor.listen(lambda event: get_lane_hist(event))

        # Lane Invasion Detector Callback
        def get_lane_hist(event):
            """
            Add timestamp of lane invasion event"""
            self.laneInv_list.append(event.timestamp)

        # Set Spectator Navigation (Location)
        self._set_spectator()

        # Wait till camera ready
        while self.cam_bp is None:
            time.sleep(0.01)

        # Update timesteps
        self.total_timesteps = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Take an action at each step"""
        # Calculate steering
        if self.discrete:
            steer = self.discrete_act[0][action % self.n_steer]
        else:
            steer = action[0]

        # Apply Control
        act = carla.VehicleControl(
            throttle=self.throttle_val, steer=steer * CarlaEnv.steer_amp
        )
        self.ego_vehicle.apply_control(act)

        # Set Spectator Navigation (Location)
        self._set_spectator()

        info = self._get_info()

        # Update timesteps
        self.total_timesteps += 1
        self.episode_steps += 1

        self.reward = self._get_reward()
        self.total_reward += self.reward
        self.episode_reward += self.reward

        self.statistics.loc[len(self.statistics)] = [  # type: ignore
            self.total_timesteps,
            self.episode_steps,
            self.scenario,
            self.total_reward,
            self.episode_reward,
            self.reward,
        ]

        return self._get_obs(), self.reward, self._terminal(), False, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def destroy_actors(self) -> None:
        """
        Destroy all actors in the actor_list"""
        for actor_filter in self.actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == "controller.ai.walker":
                        actor.stop()
                actor.destroy()
        print("\nAll Actors Destroyed")

    @staticmethod
    def _signal_handler(signal, frame):
        """Handles interrupt from Keyboard (Ctrl + C)"""

        sys.exit()

    def _set_spectator(self, transform=None) -> None:
        """
        Focuses world spectator at ego vehilce"""
        spect = self.world.get_spectator()
        if transform == None:
            transform = self.ego_vehicle.get_transform()
            transform.location.z += 3  # ABOVE vehicle
            transform.rotation.pitch -= 30  # LOOK down
        spect.set_transform(transform)

    def _get_obs(self):
        """
        Get the observations"""

        # camera = resize(self.cam_img, (self.obs_size[0], self.obs_size[1])) * 255
        # obs = {
        # 'camera':camera.astype(np.uint8),
        # }
        # obs = {
        # 'camera':self.cam_img,
        # }
        # cv2.imwrite("cam_image.jpg",  self.cam_img)
        # print(self.cam_img.shape)
        obs = self.preprocess_image(self.cam_img)
        # obs_img = np.transpose(obs, (1, 2, 0))
        # print(f"\n{obs.shape}\n")
        # print(f"\n{type(obs)}\n")
        # # obs = np.transpose(obs, (1, 2, 0))
        # # img = np.array([obs[0].numpy(), obs[1].numpy(), obs[2].numpy()])
        # img = obs_img.numpy()
        # print(np.max(img))
        # output = 0 + ((255 - 0) / (1 - -1)) * (img - -1)
        # print(type(img))
        # print(img.shape)
        # print(np.max(output))
        # cv2.imwrite("observation.jpg",  output)
        return obs
        # return obs

    def _get_reward(self):
        """
        Calculate the step reward"""
        # Reward for speed tracking (in Km/hr)
        vel = self.ego_vehicle.get_velocity()
        speed = int(3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2))
        r_speed = 0
        if speed > self.max_speed or speed < self.min_speed:
            r_speed = -1

        # Reward for collision
        r_collision = 0
        if len(self.collision_hist) != 0:
            r_collision = -1

        # Reward for lane invasion
        r_out = 0
        if len(self.laneInv_list) != 0:
            r_out = -1

        # reward = 200*r_collision + 10*r_speed + 1*r_out
        reward = r_collision + 0.2 * r_out
        return reward

    def _terminal(self):
        """
        Check whether to terminate the current episode."""
        # If collides
        if len(self.collision_hist) > 0:
            print("\nVehicle Collided")
            return True

        # # If out of lane
        # if len(self.laneInv_list) > 0:
        #     print("\nLane Invasion")
        #     return True

        # If reach maximum timestep
        if self.total_timesteps % self.max_time_episode == 0:
            print("\nTimeout")
            return True

        return False

    def _get_info(self):
        return {"Collision": self.collision_hist, "LaneInv": self.laneInv_list}
