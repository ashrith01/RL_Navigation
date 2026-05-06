"""
Training loops for Q-Learning, DQN, and PPO.

All three return a TrainingHistory with identical fields so downstream code
(evaluation, plotting) can treat every method uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from environment import GridWorldEnv
from q_learning import QLearningAgent
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent


# ── result container ──────────────────────────────────────────────────────────
@dataclass
class TrainingHistory:
    method: str
    rewards: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    successes: List[bool] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    agent: Any = field(default=None, repr=False)

    # ── derived statistics ────────────────────────────────────────────────────

    def success_rate(self, last_n: int = 50) -> float:
        tail = self.successes[-last_n:]
        return float(np.mean(tail)) if tail else 0.0

    def avg_reward(self, last_n: int = 50) -> float:
        tail = self.rewards[-last_n:]
        return float(np.mean(tail)) if tail else 0.0

    def avg_steps(self, last_n: int = 50) -> float:
        tail = [s for s, ok in zip(self.steps[-last_n:], self.successes[-last_n:]) if ok]
        return float(np.mean(tail)) if tail else float(np.mean(self.steps[-last_n:]))

    def convergence_episode(self, threshold: float = 0.9, window: int = 20) -> Optional[int]:
        """First episode at which rolling success rate exceeds threshold."""
        for ep in range(window, len(self.successes) + 1):
            if np.mean(self.successes[ep - window : ep]) >= threshold:
                return ep
        return None

    def smoothed_rewards(self, window: int = 20) -> np.ndarray:
        r = np.array(self.rewards, dtype=float)
        return np.convolve(r, np.ones(window) / window, mode="valid")


# ── Q-Learning ────────────────────────────────────────────────────────────────
def train_q_learning(
    env: GridWorldEnv,
    n_episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    seed: int = 0,
) -> TrainingHistory:
    np.random.seed(seed)

    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
    )

    hist = TrainingHistory(
        method="Q-Learning",
        config=dict(alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay),
        agent=agent,
    )

    for _ in range(n_episodes):
        env.reset()
        state = env.get_state_index()
        ep_reward, ep_steps, success = 0.0, 0, False
        done = False

        while not done:
            action = agent.select_action(state)
            _, reward, done, info = env.step(action)
            next_state = env.get_state_index()
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            ep_steps += 1
            success = info["success"]

        agent.decay_epsilon()
        hist.rewards.append(ep_reward)
        hist.steps.append(ep_steps)
        hist.successes.append(success)
        hist.losses.append(0.0)   # tabular — no neural loss

    return hist


# ── DQN ───────────────────────────────────────────────────────────────────────
def train_dqn(
    env: GridWorldEnv,
    n_episodes: int = 500,
    hidden: int = 128,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    tau: float = 0.005,
    batch_size: int = 64,
    learn_start: int = 256,
    seed: int = 0,
) -> TrainingHistory:
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)

    agent = DQNAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden=hidden,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        tau=tau,
        batch_size=batch_size,
        learn_start=learn_start,
    )

    hist = TrainingHistory(
        method="DQN",
        config=dict(lr=lr, gamma=gamma, hidden=hidden, epsilon_decay=epsilon_decay),
        agent=agent,
    )

    for _ in range(n_episodes):
        state = env.reset()
        ep_reward, ep_steps, success = 0.0, 0, False
        ep_losses: List[float] = []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss:
                ep_losses.append(loss)
            state = next_state
            ep_reward += reward
            ep_steps += 1
            success = info["success"]

        agent.decay_epsilon()
        hist.rewards.append(ep_reward)
        hist.steps.append(ep_steps)
        hist.successes.append(success)
        hist.losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

    return hist


# ── PPO ───────────────────────────────────────────────────────────────────────
def train_ppo(
    env: GridWorldEnv,
    n_episodes: int = 500,
    hidden: int = 128,
    lr: float = 2e-4,          # conservative lr prevents mode collapse
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_ratio: float = 0.1,   # tight clip → small policy steps
    value_coef: float = 0.5,
    entropy_coef: float = 0.1, # strong entropy bonus maintains exploration
    n_rollout_eps: int = 8,    # larger rollout → more diverse experience
    n_epochs: int = 3,         # fewer epochs → less overfitting per rollout
    mini_batch_size: int = 64,
    seed: int = 0,
) -> TrainingHistory:
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)

    agent = PPOAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden=hidden,
        lr=lr,
        gamma=gamma,
        lam=lam,
        clip_ratio=clip_ratio,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        n_rollout_eps=n_rollout_eps,
        n_epochs=n_epochs,
        mini_batch_size=mini_batch_size,
    )

    hist = TrainingHistory(
        method="PPO",
        config=dict(lr=lr, gamma=gamma, clip_ratio=clip_ratio, entropy_coef=entropy_coef),
        agent=agent,
    )

    rollout_count = 0
    pending_losses: List[float] = []

    for ep in range(n_episodes):
        success, ep_reward, ep_steps = agent.collect_episode(env)

        rollout_count += 1
        if rollout_count >= n_rollout_eps:
            loss = agent.update()
            pending_losses.append(loss)
            rollout_count = 0

        hist.rewards.append(ep_reward)
        hist.steps.append(ep_steps)
        hist.successes.append(success)
        hist.losses.append(pending_losses[-1] if pending_losses else 0.0)

    return hist
