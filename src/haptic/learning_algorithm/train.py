# import gym
from haptic.gym.envs.box2d.lunar_lander import LunarLander
from haptic.learning_algorithm.dqn import Agent
import numpy as np
import torch as th

if __name__ == "__main__":
    # env = gym.make("LunarLander-v2")
    env = LunarLander()
    agent = Agent(
        gamma=0.99,
        epsilon=1,
        batch_size=64,
        n_actions=4,
        eps_end=0.01,
        input_dims=[8],
        lr=0.003,
        max_mem_size=5000,
    )
    scores, eps_history = [], []
    n_games = 500
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action=action)
            score += reward
            agent.store_transitions(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print(
            "episode",
            i,
            f"score {score}",
            f"avg_score {avg_score}",
            f"epsilon {agent.epsilon}",
        )
    model = agent.Q_pred
    th.save(model, "DQN_Lunar")
