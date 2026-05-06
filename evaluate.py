"""
Policy evaluation and result aggregation.

evaluate_agent()  — run a trained RL agent for n_eval_episodes and collect metrics
rollout_rl()      — single deterministic rollout returning full path + obstacle history
evaluate_dijkstra() — run Dijkstra on the current obstacle layout, mimic RL episode
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from environment import GridWorldEnv
from dijkstra import run_dijkstra, DijkstraResult


# ── result containers ─────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    method: str
    success_rate: float
    avg_reward: float
    avg_steps: float
    avg_path_length: float    # steps on successful episodes only
    n_episodes: int
    paths: List[List[Tuple[int, int]]] = field(default_factory=list, repr=False)
    obstacle_snapshots: List[Dict] = field(default_factory=list, repr=False)
    # per-step dynamic obstacle positions for each episode (for GIF animation)
    dynamic_histories: List[List[List[Tuple[int, int]]]] = field(default_factory=list, repr=False)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "success_rate": round(self.success_rate, 4),
            "avg_reward": round(self.avg_reward, 2),
            "avg_steps": round(self.avg_steps, 2),
            "avg_path_length": round(self.avg_path_length, 2),
            "n_episodes": self.n_episodes,
        }


# ── single rollout helpers ────────────────────────────────────────────────────
def _dyn_pos(env: GridWorldEnv) -> List[Tuple[int, int]]:
    """Snapshot current dynamic obstacle positions."""
    return [tuple(p) for p in env.dynamic_pos]


def rollout_q(agent, env: GridWorldEnv, stochastic: bool = False):
    """Q-learning rollout. stochastic=True uses ε=0.4 for visual variety."""
    env.reset()
    path = [tuple(env.agent_pos)]
    snap = env.get_obstacle_snapshot()
    dyn_hist = [_dyn_pos(env)]
    ep_r, done, success = 0.0, False, False
    saved_eps = getattr(agent, "epsilon", 0)
    if stochastic:
        agent.epsilon = 0.4
    while not done:
        state = env.get_state_index()
        action = agent.select_action(state, greedy=not stochastic)
        _, reward, done, info = env.step(action)
        path.append(info["pos"])
        dyn_hist.append(_dyn_pos(env))
        ep_r += reward
        success = info["success"]
    if stochastic:
        agent.epsilon = saved_eps
    return path, snap, success, ep_r, len(path) - 1, dyn_hist


def rollout_dqn(agent, env: GridWorldEnv, stochastic: bool = False):
    """DQN rollout."""
    state = env.reset()
    path = [tuple(env.agent_pos)]
    snap = env.get_obstacle_snapshot()
    dyn_hist = [_dyn_pos(env)]
    ep_r, done, success = 0.0, False, False
    while not done:
        action = agent.select_action(state, greedy=not stochastic)
        state, reward, done, info = env.step(action)
        path.append(info["pos"])
        dyn_hist.append(_dyn_pos(env))
        ep_r += reward
        success = info["success"]
    return path, snap, success, ep_r, len(path) - 1, dyn_hist


def rollout_ppo(agent, env: GridWorldEnv, stochastic: bool = False):
    """PPO rollout."""
    state = env.reset()
    path = [tuple(env.agent_pos)]
    snap = env.get_obstacle_snapshot()
    dyn_hist = [_dyn_pos(env)]
    ep_r, done, success = 0.0, False, False
    while not done:
        action = agent.select_action(state, greedy=not stochastic)
        state, reward, done, info = env.step(action)
        path.append(info["pos"])
        dyn_hist.append(_dyn_pos(env))
        ep_r += reward
        success = info["success"]
    return path, snap, success, ep_r, len(path) - 1, dyn_hist


# ── agent evaluation ──────────────────────────────────────────────────────────
def evaluate_agent(
    method: str,
    agent,
    env: GridWorldEnv,
    n_episodes: int = 30,
    seed: int = 99,
) -> EvalResult:
    np.random.seed(seed)
    rollout_fn = {"Q-Learning": rollout_q, "DQN": rollout_dqn, "PPO": rollout_ppo}[method]

    rewards, steps_all, path_lens, paths, snaps, dyn_hists = [], [], [], [], [], []
    successes = []

    for _ in range(n_episodes):
        path, snap, success, reward, n_steps, dyn_h = rollout_fn(agent, env)
        rewards.append(reward)
        steps_all.append(n_steps)
        successes.append(success)
        paths.append(path)
        snaps.append(snap)
        dyn_hists.append(dyn_h)
        if success:
            path_lens.append(n_steps)

    return EvalResult(
        method=method,
        success_rate=float(np.mean(successes)),
        avg_reward=float(np.mean(rewards)),
        avg_steps=float(np.mean(steps_all)),
        avg_path_length=float(np.mean(path_lens)) if path_lens else float(env.max_steps),
        n_episodes=n_episodes,
        paths=paths,
        obstacle_snapshots=snaps,
        dynamic_histories=dyn_hists,
    )


# ── Dijkstra evaluation ───────────────────────────────────────────────────────
def evaluate_dijkstra(env: GridWorldEnv) -> EvalResult:
    """Run Dijkstra on the static obstacle layout (single deterministic run)."""
    obs = set(env.static_obstacles)
    result: DijkstraResult = run_dijkstra(
        grid_size=(env.H, env.W),
        obstacles=obs,
        start=env.start,
        goal=env.goal,
    )
    path = result.path if result.success else []
    reward = result.reward if result.success else -float(env.max_steps)
    steps = result.steps if result.success else env.max_steps

    return EvalResult(
        method="Dijkstra",
        success_rate=1.0 if result.success else 0.0,
        avg_reward=reward,
        avg_steps=float(steps),
        avg_path_length=float(steps) if result.success else float(env.max_steps),
        n_episodes=1,
        paths=[path],
        obstacle_snapshots=[env.get_obstacle_snapshot()],
    )


def dijkstra_explored(env: GridWorldEnv) -> DijkstraResult:
    """Return full Dijkstra result including exploration order (for animation)."""
    return run_dijkstra(
        grid_size=(env.H, env.W),
        obstacles=set(env.static_obstacles),
        start=env.start,
        goal=env.goal,
    )


# ── CSV / JSON export ─────────────────────────────────────────────────────────
def write_eval_csv(results: List[EvalResult], path: str) -> None:
    if not results:
        return
    rows = [r.as_dict() for r in results]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_json(data: Any, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
