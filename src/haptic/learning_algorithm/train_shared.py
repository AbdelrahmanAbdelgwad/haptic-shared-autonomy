from haptic.gym.envs.box2d.lunar_lander import LunarLanderShared
from haptic.learning_algorithm.shared_dqn import Agent
import numpy as np
import torch as th
import matplotlib.pyplot as plt

LOAD_MODEL = True
MAX_EPISODE_STEPS = 500
if __name__ == "__main__":
    env = LunarLanderShared(max_episode_steps=MAX_EPISODE_STEPS)
    agent = Agent(
        gamma=0.99,
        epsilon=1,
        batch_size=64,
        n_actions=4,
        eps_end=0.01,
        input_dims=[9],
        lr=0.003,
        max_mem_size=5000,
        alpha=0.4,
    )
    if LOAD_MODEL:
        model = th.load("DQN_Lunar_Shared_alpha_0.4_with_pretrained_model_as_pilot_5")
        agent.Q_pred = model
        print("\n model loaded successfully \n")
    scores, eps_history, avg_scores = [], [], []
    n_games = 500
    total_steps = 0
    pilot = th.load("DQN_Lunar")
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        episode_steps = 0
        while not done:
            # if episode_steps >= 500:
            #     break
            episode_steps += 1
            # pi_action = env.action_space.sample()
            state = th.tensor(observation[:8]).to(agent.Q_pred.device)
            pi_q_values = pilot.forward(state).cpu().data.numpy()
            pi_action = np.argmax(pi_q_values).item()
            observation[8] = pi_action
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(
                action=action, pi_action=pi_action
            )
            score += reward
            agent.store_transitions(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        total_steps += episode_steps

        print(
            "episode",
            i,
            f"score {score}",
            f"avg_score {avg_score}",
            f"epsilon {agent.epsilon}",
            f"episode_steps {episode_steps}",
            f"total_steps {total_steps}",
        )
        model = agent.Q_pred
        th.save(model, "DQN_Lunar_Shared_alpha_0.4_with_pretrained_model_as_pilot_6")

        # build the plot
        plt.plot(avg_scores)
        plt.xlabel("timesteps")
        plt.ylabel("average score")
        plt.title("average score during training")
        # plt.show()
        plt.savefig(f"training_using_alpha_0.4_6.png")
        # plt.close()
