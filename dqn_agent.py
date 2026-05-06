"""
Dueling Double DQN with Prioritized Experience Replay (PER).

Architecture  : shared trunk → value stream V(s) + advantage stream A(s,a)
                Q(s,a) = V(s) + (A(s,a) − mean_a A(s,a))

Double DQN    : online net selects action; target net evaluates it
                → eliminates maximisation bias

PER (α=0.6)   : sample proportional to |TD-error|^α via a SumTree
IS weights    : correct sampling bias during gradient updates

Soft updates  : θ_target ← τ·θ_online + (1−τ)·θ_target  (Polyak, τ=0.005)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ── device helper ─────────────────────────────────────────────────────────────
def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── SumTree (for O(log N) priority sampling) ──────────────────────────────────
class _SumTree:
    """Binary tree where each leaf holds a priority; internal nodes sum children."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: List = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        return self._retrieve(left, s) if s <= self.tree[left] else self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        leaf = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(leaf, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def get(self, s: float) -> Tuple[int, float, object]:
        leaf = self._retrieve(0, s)
        data_idx = leaf - self.capacity + 1
        return leaf, float(self.tree[leaf]), self.data[data_idx]


# ── Prioritized Replay Buffer ──────────────────────────────────────────────────
class PrioritizedReplayBuffer:
    """
    Parameters
    ----------
    capacity   : max transitions stored
    alpha      : priority exponent (0 = uniform, 1 = fully prioritised)
    beta_start : IS weight exponent start (annealed to 1.0)
    beta_steps : steps to anneal beta from beta_start → 1.0
    """

    def __init__(
        self,
        capacity: int = 50_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int = 100_000,
        eps: float = 1e-6,
    ) -> None:
        self.tree = _SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_inc = (1.0 - beta_start) / beta_steps
        self.eps = eps
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done) -> None:
        transition = (state, int(action), float(reward), next_state, bool(done))
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size: int):
        indices, priorities, batch = [], [], []
        segment = self.tree.total / batch_size
        self.beta = min(1.0, self.beta + self.beta_inc)

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, pri, data = self.tree.get(s)
            if data is None:
                continue
            indices.append(idx)
            priorities.append(pri)
            batch.append(data)

        if not batch:
            return None

        probs = np.array(priorities) / self.tree.total
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors):
            priority = (abs(float(err)) + self.eps) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.tree.n_entries


# ── Neural Network ─────────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    """
    Dueling network: shared trunk → separate value and advantage heads.
    Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
        )
        h2 = hidden // 2
        self.value_head = nn.Sequential(nn.Linear(h2, 32), nn.ReLU(), nn.Linear(32, 1))
        self.adv_head = nn.Sequential(nn.Linear(h2, 32), nn.ReLU(), nn.Linear(32, n_actions))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.trunk(x)
        V = self.value_head(f)
        A = self.adv_head(f)
        return V + (A - A.mean(dim=1, keepdim=True))


# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    Parameters
    ----------
    state_dim      : input feature dimension (12 for GridWorldEnv)
    n_actions      : number of discrete actions
    hidden         : neurons per hidden layer
    lr             : Adam learning rate
    gamma          : discount factor
    epsilon        : initial exploration rate
    epsilon_decay  : per-episode multiplicative decay
    epsilon_min    : floor for epsilon
    tau            : Polyak soft-update coefficient
    batch_size     : replay batch size
    learn_start    : transitions stored before first gradient step
    buffer_kw      : kwargs forwarded to PrioritizedReplayBuffer
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        tau: float = 0.005,
        batch_size: int = 64,
        learn_start: int = 256,
        **buffer_kw,
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.batch_size = batch_size
        self.learn_start = learn_start

        self.device = _device()

        self.online = DuelingDQN(state_dim, n_actions, hidden).to(self.device)
        self.target = DuelingDQN(state_dim, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(**buffer_kw) if buffer_kw else PrioritizedReplayBuffer()

        self._steps = 0

    # ── action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            return int(self.online(s).argmax(dim=1).item())

    # ── experience storage ────────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    # ── learning step ─────────────────────────────────────────────────────────

    def learn(self) -> float:
        if len(self.buffer) < self.learn_start:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        if batch is None:
            return 0.0

        states, actions, rewards, next_states, dones, indices, weights = batch

        S = torch.from_numpy(states).to(self.device)
        A = torch.from_numpy(actions).long().to(self.device)
        R = torch.from_numpy(rewards).to(self.device)
        S2 = torch.from_numpy(next_states).to(self.device)
        D = torch.from_numpy(dones).to(self.device)
        W = torch.from_numpy(weights).to(self.device)

        with torch.no_grad():
            # Double DQN: online selects action, target evaluates
            best_a = self.online(S2).argmax(dim=1, keepdim=True)
            q_next = self.target(S2).gather(1, best_a).squeeze(1)
            target = R + self.gamma * q_next * (1.0 - D)

        q_pred = self.online(S).gather(1, A.unsqueeze(1)).squeeze(1)
        td_errors = (target - q_pred).detach().cpu().numpy()

        loss = (W * F.huber_loss(q_pred, target, reduction="none")).mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        self.buffer.update_priorities(indices, td_errors)
        self._soft_update()
        self._steps += 1

        return float(loss.item())

    def _soft_update(self) -> None:
        for p_o, p_t in zip(self.online.parameters(), self.target.parameters()):
            p_t.data.mul_(1.0 - self.tau).add_(p_o.data * self.tau)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
