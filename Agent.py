import torch
import numpy as np
import csv
import gymnasium as gym
import hashlib

class PkAgent:
    def __init__(self, state_dim, action_dim, save_dir, env, max_x, max_y):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.env = env

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.curr_episode = 0
        self.save_every = int(5e5)
        self.gamma = 0.9
        self.burnin = int(1e4)
        self.learn_every = 3
        self.sync_every = int(1e4)

        self.training_error = []

        self.lr = 0.01
        self.discount_factor = 0.9

        self.epsilon = 1
        self.epsilon_decay = 0.99999975
        self.final_epsilon = 0.1

        use_cuda = torch.cuda.is_available()
        print(f"Using CUDA: {use_cuda}")
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.max_x = max_x
        self.max_y = max_y
        self.num_states = max_x * max_y * 2

        self.q_values = torch.zeros((self.num_states, action_dim), device=self.device)

    def _state_to_index(self, state):
        state_bytes = bytes(state)  # must be a list of ints
        hash_val = hashlib.sha256(state_bytes).hexdigest()
        return int(hash_val, 16) % self.num_states

    def load(self, path):
        self.q_values = torch.load(path)

    def save(self, path):
        torch.save(self.q_values, f"{path}/pk_net_{self.curr_episode}.pt")

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        idx = self._state_to_index(obs)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(torch.argmax(self.q_values[idx]).item())

    def update(self, obs, action, reward, terminated, next_obs):
        idx = self._state_to_index(obs)
        next_idx = self._state_to_index(next_obs)

        future_q = 0.0 if terminated else torch.max(self.q_values[next_idx]).item()
        td_error = reward + self.discount_factor * future_q - self.q_values[idx, action].item()

        self.q_values[idx, action] += self.lr * td_error
        self.training_error.append(td_error)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)