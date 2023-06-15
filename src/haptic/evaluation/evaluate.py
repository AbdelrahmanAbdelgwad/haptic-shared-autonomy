import gym
from gym import logger as gymlogger

gymlogger.set_level(30)
import glob
import io
import os
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay

from stable_baselines3 import DQN
from haptic.envs.car_racing import CarRacingDiscrete

# import wandb


def evaluate_version(model, env, version_name, version, video_path):
    # get version of model
    MODEL_SAVE_NAME = version_name
    SAVED_MODEL_VERSION = version
    # # wnadb api key: 00d5bfbd342bb73d5aaf4f2833436d20457ef040
    # os.environ["WANDB_ENTITY"] = "andreas_giannoutsos"
    # os.environ["WANDB_PROJECT"] = "gym_car_racer"
    # os.environ["WANDB_RESUME"] = "allow"
    # wandb.init()
    # model_artifact = wandb.use_artifact(
    #     MODEL_SAVE_NAME + ":" + SAVED_MODEL_VERSION, type="model"
    # )
    # artifact_dir = model_artifact.download()
    # loaded_model = model.load(artifact_dir + "/" + MODEL_SAVE_NAME)

    loaded_model = DQN.load(MODEL_SAVE_NAME)
    # play model
    env = gym.wrappers.Monitor(env, video_path, force=True)
    obs = env.reset()
    done = False
    while not done:
        action, _states = loaded_model.predict(obs.copy(), deterministic=True)
        obs, reward, done, info = env.step(action)
    env.close()

    # display video
    mp4list = glob.glob(video_path + "/*.mp4")
    print(mp4list)
    if len(mp4list) > 0:
        print(len(mp4list))
        mp4 = max(mp4list, key=os.path.getctime)
        # mp4 = mp4list[-1]
        video = io.open(mp4, "r+b").read()
        encoded = base64.b64encode(video)

        # log gameplay video in wandb
        # wandb.log({"gameplays": wandb.Video(mp4, fps=4, format="gif")})

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
        print("Video path: ", mp4)


env = CarRacingDiscrete(1000)

# Model with 10.000 steps has best performance at 566 episode (566/20~=28)
evaluate_version(DQN, env, "DQN_model", "v28", "./videoo")

# Model with 2.000 steps has best performance at 798 episode (800/50~=15)
# evaluate_version(DQN, env, "DQN_RL_48", "v15", "./videoo")

# Model with 5.000 steps has best performance at last episode
# evaluate_version(DQN, env, "DQN_RL_54", "v9", "./videoo")
