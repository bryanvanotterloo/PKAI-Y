from collections import defaultdict
import gymnasium as gym
import numpy as np

class PkAgent:
    def __init__(self, state_dim, action_dim, save_dir,env):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.curr_episode = 0
        self.save_every = 5e5  # no. of experiences between saving
        self.gamma = 0.9
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online
        
        self.training_error = []

        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = 0.01
        self.discount_factor = 0.9

        self.epsilon = 1
        self.epsilon_decay = 0.99999975
        self.final_epsilon = 0.1

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        tup_obs = tuple(obs.flatten())
        tup_next_obs = tuple(next_obs.flatten())
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[tup_next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[tup_obs][action]
        )

        self.q_values[tup_obs][action] = (
            self.q_values[tup_obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)