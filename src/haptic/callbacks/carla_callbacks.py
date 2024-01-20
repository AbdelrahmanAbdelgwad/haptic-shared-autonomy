from typing import Optional
import time
import logging
import os
import numpy as np
from torch import nn
import gym
from pandas import DataFrame, concat
import simple_colors
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class SaveBestModelCallback(BaseCallback):
    """Runs evaluation episodes on the trained model,save the evaluation logs
    and saves the model if improved in evaluation.

    Args:
        BaseCallback (Class): Base class for callback in stable_baselines3.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        n_eval_episodes=5,
        logpath: Optional[str] = None,
        savepath: Optional[str] = None,
        eval_frequency: int = 1000,  # 50_000
        verbose: int = 1,
        render=True,
    ) -> None:
        super().__init__(verbose)

        self.logpath = logpath
        self.n_eval_episodes = n_eval_episodes
        self.eval_frequency = eval_frequency
        self.last_len_statistics = 0
        self.best_avg_reward = [-np.inf]
        self.eval_env = eval_env
        self.savepath = savepath
        self.last_eval_time = time.time()
        self.render = render

    def _on_step(self) -> bool:
        """Runs evaluation episodes on the trained model every eval_frequency timesteps,
        saves the evaluation logs and saves the model if improved in evaluation.

        Returns:
            bool: If the callback returns False, training is aborted early.
        """

        if self.eval_frequency > 0 and self.n_calls % self.eval_frequency == 0:
            tic = time.time()
            eval_logs = run_n_episodes(self.model, self.eval_env, self.n_eval_episodes)
            toc = time.time()
            eval_duration = toc - tic
            last_added_eval_logs = eval_logs[self.last_len_statistics :]
            new_avg_reward = np.mean(last_added_eval_logs["reward"].values)

            save_logs(eval_logs, self.logpath, self.verbose)
            if self.verbose:
                print_statistics_eval(last_added_eval_logs, eval_duration)
            self.save_model_if_improved(new_avg_reward, self.model, self.savepath)

            self.last_len_statistics = len(eval_logs)
        return True

    def save_model_if_improved(
        self,
        new_avg_reward: float,
        model: nn.Module,
        savepath: Optional[str],
    ) -> None:
        """Save the model if the average reward improved in evaluation.

        Args:
            new_avg_reward (float): New average reward in evaluation
            model (_type_): Trained model instance
            savepath (str): Path to save the model
        """
        if new_avg_reward <= self.best_avg_reward[0]:
            return

        self.best_avg_reward[0] = new_avg_reward
        if savepath is not None:
            try:
                model.save(savepath)
                stat_message = (
                    f"Model saved to {savepath} (avg reward: {new_avg_reward})."
                )
                print(simple_colors.green(stat_message))
            except AttributeError:
                print(simple_colors.red("Could not save"))
            # else:
            #     print(simple_colors.red("An error occured while saving the model"))
        else:
            print(simple_colors.red("No save path found"))


def save_logs(training_logs: DataFrame, logpath: Optional[str], verbose: int) -> None:
    """Saves the training logs of the robot.

    Args:
        training_logs (DataFrame): Contains robot training logs.
        logpath (str): Path to save the logs to it.
        verbose (int): Flag to print the saving path.
    """
    training_logs.to_csv(logpath)
    if verbose > 1:
        print(simple_colors.green(f"Training logs saved to {logpath}"))


def run_n_episodes(
    model: nn.Module,
    env: gym.Env,
    num_eposides: int,
) -> DataFrame:
    """Run n evaluation-episodes on the trained model.

    Args:
        model (torch.nn.Module): Trained model.
        env (gym.Env): Evaluation env.
        num_eposides (int): Number of episodes to evaluate on.

    Returns:
        DataFrame: Logs for evaluation episodes.
    """
    env.statistics["scenario"] = "robot_env_test"
    for episode in range(num_eposides):
        obs = env.reset()
        done = False
        test_steps = 0
        while not done:
            test_steps += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if test_steps == 0:
                print(simple_colors.blue(f"test episode {episode}"))
            if done:
                print(simple_colors.blue(f"test episode {episode}"))
        print(simple_colors.blue(env.statistics))
    return env.statistics


def print_statistics_eval(
    eval_logs: DataFrame,
    eval_elapsed: float,
) -> None:
    """Print statistics of evaluation

    Args:
        eval_logs (DataFrame): ontains robot evaluation logs.
        eval_elapsed (float):  time elapsed in evaluation.
    """
    scenarios = sorted(list(set(eval_logs["scenario"].values)))
    rewards = eval_logs["reward"].values
    print(simple_colors.blue(f"Evaluation time : {round(eval_elapsed, 4)}"))
    for scenario in scenarios:
        is_scenario = eval_logs["scenario"].values == scenario
        scenario_rewards = rewards[is_scenario]
        avg_scenario_rewards = np.mean(scenario_rewards).item()
        num_scenarios = len(scenario_rewards)
        stats = f"{scenario}: {avg_scenario_rewards:.4f} ({num_scenarios})"
        print(simple_colors.green(stats))


class PeriodicSaveModelCallback(BaseCallback):
    """Runs evaluation episodes on the trained model,save the evaluation logs
    and saves the model if improved in evaluation.

    Args:
        BaseCallback (Class): Base class for callback in stable_baselines3.
    """

    def __init__(
        self,
        save_path: Optional[str] = None,
        save_frequency: int = 10_000,  # 50_000
    ) -> None:
        super().__init__()

        self.save_frequency = save_frequency
        self.save_path = save_path
        # create folder if not exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _on_step(self) -> bool:
        """Saves the model every save_frequency timesteps.

        Returns:
            bool: If the callback returns False, training is aborted early.
        """
        if self.save_frequency > 0 and self.n_calls % self.save_frequency == 0:
            # Join the current save path with the current number of call
            save_path = os.path.join(self.save_path, f"model_{self.n_calls}")
            self.model.save(save_path)
            if self.verbose:
                print(f"latest model saved to {self.save_path}", simple_colors.green)
        return True
