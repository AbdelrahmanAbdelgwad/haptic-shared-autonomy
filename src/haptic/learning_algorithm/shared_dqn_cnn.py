import torch as th
from torch import nn
from torch import flatten
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tensorflow as tf

def steering2action(action):
    if action == 0:
        action = 0  # "NOTHING"
    if action == -0.2:
        action = 1  # LEFT_LEVEL_1
    if action == -0.4:
        action = 2  # LEFT_LEVEL_2
    if action == -0.6:
        action = 3  # LEFT_LEVEL_3
    if action == -0.8:
        action = 4  # LEFT_LEVEL_4
    if action == -1:
        action = 5  # LEFT_LEVEL_5
    if action == 0.2:
        action = 6  # RIGHT_LEVEL_1
    if action == 0.4:
        action = 7  # RIGHT_LEVEL_2
    if action == 0.6:
        action = 8  # RIGHT_LEVEL_3
    if action == 0.8:
        action = 9  # RIGHT_LEVEL_4
    if action == 1:
        action = 10  # RIGHT_LEVEL_5
    return action

def steering2action(action):
    if action == 0:
        action = 0  # "NOTHING"
    if action == -0.2:
        action = 1  # LEFT_LEVEL_1
    if action == -0.4:
        action = 2  # LEFT_LEVEL_2
    if action == -0.6:
        action = 3  # LEFT_LEVEL_3
    if action == -0.8:
        action = 4  # LEFT_LEVEL_4
    if action == -1:
        action = 5  # LEFT_LEVEL_5
    if action == 0.2:
        action = 6  # RIGHT_LEVEL_1
    if action == 0.4:
        action = 7  # RIGHT_LEVEL_2
    if action == 0.6:
        action = 8  # RIGHT_LEVEL_3
    if action == 0.8:
        action = 9  # RIGHT_LEVEL_4
    if action == 1:
        action = 10  # RIGHT_LEVEL_5
    return action


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, observation_space, cuda_index=0):
        super().__init__()
        self.n_actions = n_actions
        n_input_channels = observation_space.shape[-1]
        # initialize first set of CONV => RELU => POOL layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self.cnn(
                th.permute(
                    th.as_tensor(observation_space.sample()[None]).float(), (0, 3, 1, 2)
                )
            ).shape[1]

        # initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(in_features=n_flatten, out_features=512)
        self.relu3 = nn.ReLU()
        # initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=256, out_features=n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = th.device(
            f"cuda:{cuda_index}" if th.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def forward(self, state):
        if len(state.shape) == 4:
            x = th.permute(state, (0, 3, 1, 2))
            x = self.cnn(x.float())
        else:
            x = th.permute(state, (2, 0, 1))
            x = self.cnn(x.float())
            x = th.flatten(x, 0)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        actions = self.fc3(x)
        return actions


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        batch_size,
        n_actions,
        observation_space,
        max_mem_size=100000,
        eps_end=0.01,
        eps_dec=5e-4,
        max_q_target_iter=300,
        alpha=0.6,
        cuda_index=0,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.target_cntr = 0
        self.C = max_q_target_iter
        self.alpha = alpha

        self.Q_pred = DeepQNetwork(
            lr=self.lr,
            n_actions=n_actions,
            observation_space=observation_space,
            cuda_index=cuda_index,
        )
        self.Q_target = self.Q_pred

        # it is very important to assign data types when working with PyTorch as it
        # enforces some form of type checking
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool8)

    def store_transitions(self, state, action, reward, state_, done):
        index = (
            self.mem_cntr % self.mem_size
        )  # Here we are using the % operator to see if we passed
        # the allowed memory size or not. If we did, then we will start from zero, and hence rewriting
        # the memory all over again
        # we could have simply used a queue instead of named arrays but it is simpler this way
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1
        self.target_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            observation = th.tensor(observation).to(self.Q_pred.device)
            q_values = self.Q_pred.forward(observation).cpu().data.numpy()
            # q_values -= tf.reduce_min(q_values, axis=1)
            q_values -= tf.reduce_min(q_values)
            opt_action = np.argmax(q_values).item()
            # opt_q_values = q_values[0][opt_action]
            opt_q_values = q_values[opt_action]
            # pi_action = int(observation[8])
            pi_action_steering = observation.cpu().data.numpy()[:, :, -1][0][0]
            pi_action = steering2action(pi_action_steering)
            # pi_act_q_values = q_values[0][pi_action]
            pi_act_q_values = q_values[pi_action]

            if pi_act_q_values >= (1 - self.alpha) * opt_q_values:
                action = pi_action
            else:
                action = opt_action
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        self.Q_pred.optimizer.zero_grad()  # This is particular to PyTorch
        batch_selection_max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(
            batch_selection_max_mem, self.batch_size, replace=False
        )  # It is important to make replace=False so that once you select an experience, you do not again
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = th.tensor(self.state_memory[batch]).to(self.Q_pred.device)
        new_state_batch = th.tensor(self.new_state_memory[batch]).to(self.Q_pred.device)
        reward_batch = th.tensor(self.reward_memory[batch]).to(self.Q_pred.device)
        terminal_batch = th.tensor(self.terminal_memory[batch]).to(self.Q_pred.device)

        action_batch = self.action_memory[batch]

        q_pred = self.Q_pred.forward(state_batch)[batch_index, action_batch]
        if self.target_cntr > self.C:
            self.target_cntr = 0
            self.Q_target = self.Q_pred
        q_next = self.Q_target.forward(new_state_batch)
        q_next[terminal_batch] = 0
        q_target = reward_batch + self.gamma * th.max(q_next, dim=1)[0]

        loss = self.Q_pred.loss(q_target, q_pred).to(self.Q_pred.device)
        loss.backward()
        self.Q_pred.optimizer.step()

        self.epsilon = (
            (self.epsilon - self.eps_dec)
            if self.epsilon > self.eps_min
            else self.eps_min
        )
