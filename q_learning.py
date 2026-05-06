"""
Tabular Q-Learning agent.

Update rule:  Q[s,a] += α · (r + γ · max_a' Q[s',a'] − Q[s,a])
Exploration : ε-greedy with exponential decay per episode
"""

from __future__ import annotations

import numpy as np


class QLearningAgent:
    """
    Parameters
    ----------
    n_states       : number of discrete states (H*W for a grid)
    n_actions      : number of actions
    alpha          : learning rate
    gamma          : discount factor
    epsilon        : initial exploration probability
    epsilon_decay  : multiplicative decay applied after each episode
    epsilon_min    : floor for epsilon
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.Q: np.ndarray = np.zeros((n_states, n_actions), dtype=np.float64)

    # ── action selection ──────────────────────────────────────────────────────

    def select_action(self, state: int, greedy: bool = False) -> int:
        if greedy or np.random.rand() > self.epsilon:
            return int(np.argmax(self.Q[state]))
        return int(np.random.randint(self.n_actions))

    # ── Q-table update ────────────────────────────────────────────────────────

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        target = reward
        if not done:
            target += self.gamma * float(np.max(self.Q[next_state]))
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── greedy policy ─────────────────────────────────────────────────────────

    def greedy_policy(self) -> np.ndarray:
        """Return the greedy action for every state (shape: n_states)."""
        return np.argmax(self.Q, axis=1).astype(np.int32)
