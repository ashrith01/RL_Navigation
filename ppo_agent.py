"""
Proximal Policy Optimisation (PPO) with Generalised Advantage Estimation (GAE).

Architecture  : shared MLP trunk → policy head (softmax) + value head (scalar)
Rollout       : collect n_rollout_eps complete episodes, then update K epochs
GAE           : λ-weighted advantage for low-variance gradient estimates
Clipped loss  : L = -min(r_t·A, clip(r_t, 1-ε, 1+ε)·A)  — prevents destructive updates
Entropy bonus : encourages continued exploration throughout training
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple


# ── device helper ─────────────────────────────────────────────────────────────
def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Actor-Critic network ──────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    """Shared-trunk Actor-Critic network."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

        # Orthogonal initialisation — standard for PPO
        for m in self.shared.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def act(self, state: np.ndarray, greedy: bool = False) -> Tuple[int, float, float]:
        """Sample action, return (action, log_prob, value)."""
        with torch.no_grad():
            x = torch.from_numpy(state).float().unsqueeze(0).to(next(self.parameters()).device)
            logits, value = self.forward(x)
            dist = Categorical(logits=logits)
            action = logits.argmax(dim=-1) if greedy else dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())


# ── PPO Agent ─────────────────────────────────────────────────────────────────
class PPOAgent:
    """
    Parameters
    ----------
    state_dim         : input dimension
    n_actions         : discrete action count
    hidden            : neurons per hidden layer
    lr                : Adam learning rate
    gamma             : discount factor
    lam               : GAE λ (bias-variance trade-off, 0.95 is typical)
    clip_ratio        : PPO clipping parameter ε (0.2 is standard)
    value_coef        : weight for value function loss
    entropy_coef      : weight for entropy bonus (encourages exploration)
    n_rollout_eps     : episodes collected before each gradient update
    n_epochs          : gradient epochs per rollout
    mini_batch_size   : mini-batch size within each epoch
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_rollout_eps: int = 8,
        n_epochs: int = 4,
        mini_batch_size: int = 64,
    ) -> None:
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_rollout_eps = n_rollout_eps
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size

        self.device = _device()
        self.net = ActorCritic(state_dim, n_actions, hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self._rollout_buffer: List = []

    # ── episode data collection ────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        action, _, _ = self.net.act(state, greedy=greedy)
        return action

    def collect_episode(self, env) -> Tuple[bool, float, int]:
        """Run one episode, append transitions to rollout buffer."""
        state = env.reset()
        ep_reward, ep_steps, success = 0.0, 0, False
        done = False
        while not done:
            action, log_prob, value = self.net.act(state)
            next_state, reward, done, info = env.step(action)
            self._rollout_buffer.append((state, action, reward, done, log_prob, value))
            state = next_state
            ep_reward += reward
            ep_steps += 1
            success = info.get("success", False)
        return success, ep_reward, ep_steps

    def ready_to_update(self) -> bool:
        return len(self._rollout_buffer) > 0 and (
            len(self._rollout_buffer) % self.n_rollout_eps == 0
            or len(self._rollout_buffer) >= self.n_rollout_eps
        )

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(self) -> float:
        if not self._rollout_buffer:
            return 0.0

        buf = self._rollout_buffer
        self._rollout_buffer = []

        states, actions, rewards, dones, old_log_probs, values = (
            [b[i] for b in buf] for i in range(6)
        )
        states_arr = np.array(states, dtype=np.float32)
        actions_arr = np.array(actions, dtype=np.int64)
        rewards_arr = np.array(rewards, dtype=np.float32)
        rewards_arr = rewards_arr / 100.0  # scale to [-1,1] for stable value learning
        dones_arr = np.array(dones, dtype=np.float32)
        old_lp_arr = np.array(old_log_probs, dtype=np.float32)
        values_arr = np.array(values, dtype=np.float32)

        advantages, returns = self._compute_gae(rewards_arr, dones_arr, values_arr)

        # Normalise advantages
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Convert to tensors
        S = torch.from_numpy(states_arr).to(self.device)
        A = torch.from_numpy(actions_arr).long().to(self.device)
        R = torch.from_numpy(returns.astype(np.float32)).to(self.device)
        ADV = torch.from_numpy(advantages.astype(np.float32)).to(self.device)
        OLD_LP = torch.from_numpy(old_lp_arr).to(self.device)

        T = len(S)
        total_loss = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            perm = torch.randperm(T, device=self.device)
            for start in range(0, T, self.mini_batch_size):
                idx = perm[start : start + self.mini_batch_size]
                if idx.numel() < 2:
                    continue

                logits, vals = self.net(S[idx])
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(A[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - OLD_LP[idx])
                adv_b = ADV[idx]
                clip = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                pol_loss = -torch.min(ratio * adv_b, clip * adv_b).mean()
                val_loss = F.mse_loss(vals, R[idx])

                loss = pol_loss + self.value_coef * val_loss - self.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()

                total_loss += float(loss.item())
                n_updates += 1

        return total_loss / max(n_updates, 1)

    # ── GAE ───────────────────────────────────────────────────────────────────

    def _compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float64)
        gae = 0.0
        for t in reversed(range(T)):
            nv = 0.0 if dones[t] else (values[t + 1] if t + 1 < T else 0.0)
            delta = rewards[t] + self.gamma * nv - values[t]
            gae = delta + self.gamma * self.lam * (0.0 if dones[t] else gae)
            adv[t] = gae
        returns = adv + values
        return adv, returns
