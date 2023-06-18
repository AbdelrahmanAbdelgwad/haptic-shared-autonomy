import torch

# import gnwrapper
from haptic.gym import logger as gymlogger

from haptic.gym.wrappers import Monitor

# from gym.wrappers.record_video import RecordVideo

gymlogger.set_level(30)
import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay

# from haptic.envs.car_racing import CarRacingDiscrete
from haptic.gym.envs.box2d.car_racing import CarRacing
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN
from stable_baselines3.dqn import CnnPolicy

NUM = 54
NUM_OF_STEPS = 1_000_000
NUM_OF_EPISODES = 1
LOG_INTERVAL = 50
BUFFER_SIZE = 50000
LEARNING_STARTS = 50000
MODEL_SAVE_NAME = "DQN_RL_" + str(NUM)
SAVED_MODEL_VERSION = "latest"
LOAD_SAVED_MODEL = False


class DQNCustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(DQNCustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.episodes = 0
        self.total_episode_reward = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        # update commulative reward to log at the end of every episode
        self.total_episode_reward += self.locals["reward"]
        # at the end of every episode
        if self.locals["done"][0]:
            # if log interval has passed
            if self.episodes % self.locals["log_interval"] == 0:
                # Print the last video
                # Save your model and optimizer
                self.model.save(MODEL_SAVE_NAME)
                mp4list = glob.glob("video/*.mp4")
                print(mp4list)
                if len(mp4list) > 0:
                    print(len(mp4list))
                    mp4 = mp4list[-1]
                    video = io.open(mp4, "r+b").read()
                    encoded = base64.b64encode(video)

                    # display gameplay video
                    ipythondisplay.clear_output(wait=True)
                    ipythondisplay.display(
                        HTML(
                            data="""<video alt="" autoplay 
                                    loop controls style="height: 400px;">
                                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                                </video>""".format(
                                encoded.decode("ascii")
                            )
                        )
                    )
                    print("Episode:", self.episodes)
            self.episodes += 1
            self.total_episode_reward = 0

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def add(x, y):
    return x + y


def main():
    # if gpu is to be used
    # pylint: disable=E1101
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pylint: enable=E1101
    print("using", device)

    # env = gnwrapper.Animation(CarRacingDiscrete())
    # env = CarRacingDiscrete()
    env = CarRacing(
        allow_reverse=False,
        grayscale=1,
        show_info_panel=1,
        discretize_actions="soft",
        num_tracks=2,
        num_lanes=2,
        num_lanes_changes=4,
        max_time_out=2,
        frames_per_state=4,
    )
    # print(env.action_space.shape)
    # print(env.action_space)

    # env.reset()
    # env.render()
    # im = env.render("state_pixels")

    # def state_image_preprocess(state_image):
    #     state_image = state_image.transpose((2, 0, 1))
    #     state_image = np.ascontiguousarray(state_image, dtype=np.float32) / 255
    #     state_image = torch.from_numpy(state_image)
    #     return state_image.unsqueeze(0).to(device)

    # state_image_preprocess(im).shape
    # plt.imshow(state_image_preprocess(im).cpu().squeeze(0).permute(1, 2, 0).numpy())

    # Use wrappers.Monitor in order to have a video
    # env = RecordVideo(CarRacingDiscrete(NUM_OF_STEPS),'./video',  episode_trigger = lambda episode_number: True)
    # env = CarRacingDiscrete(NUM_OF_STEPS)
    # Load model
    if LOAD_SAVED_MODEL:
        DQN.load("DQN_model", env=env)
    else:
        if "DQNmodel" not in globals():
            DQNmodel = DQN(
                CnnPolicy,
                env,
                verbose=2,
                buffer_size=BUFFER_SIZE,
                learning_starts=LEARNING_STARTS,
            )
            print("INITIALIZE NEW DQN MODEL")
        else:
            DQNmodel = DQN.load(MODEL_SAVE_NAME, env=env)
            print("CONTINUE DQN MODEL TRAINING")

    t1 = time.time()
    # Train model
    DQNmodel.learn(
        total_timesteps=NUM_OF_STEPS * NUM_OF_EPISODES,
        log_interval=LOG_INTERVAL,
    )
    t2 = time.time()
    dt = t2 - t1
    time_in_hours = ((dt / 1000) / 60) / 60
    print("\n", "training time was", time_in_hours, "\n")
    # Save model
    DQNmodel.save("DQN_model_hard_actions")


if __name__ == "__main__":
    main()
