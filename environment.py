"""
GridWorldEnv — 2-D robot navigation environment.

State  : 12-dim float32 vector
           [agent_r/H, agent_c/W,        (normalised position)
            (goal_r-r)/H, (goal_c-c)/W,  (goal direction)
            d_N, d_S, d_W, d_E,          (cardinal sensors)
            d_NW, d_NE, d_SW, d_SE]      (diagonal sensors)
           Each sensor = normalised steps to nearest obstacle / wall (0=adjacent, 1=clear)

Action : 0=Up  1=Down  2=Left  3=Right
Reward : +100 reach goal | -1 per step | -10 obstacle collision
         optional potential-based shaping: +0.5*(prev_dist - curr_dist)
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, FrozenSet, List, Set, Tuple

# ── action / sensor constants ─────────────────────────────────────────────────
ACTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES: List[str] = ["Up", "Down", "Left", "Right"]

SENSOR_DIRS: List[Tuple[int, int]] = [
    (-1, 0), (1, 0), (0, -1), (0, 1),      # N S W E
    (-1, -1), (-1, 1), (1, -1), (1, 1),     # NW NE SW SE
]
STATE_DIM: int = 12


# ── main class ────────────────────────────────────────────────────────────────
class GridWorldEnv:
    """
    Parameters
    ----------
    grid_size        : (rows, cols)
    n_obstacles      : total obstacle count
    seed             : RNG seed for reproducible obstacle layout
    dynamic_fraction : fraction [0,1] of obstacles that move each step
    motion_mode      : 'patrol' (bounce) | 'homing' (chase agent) | 'random'
    shaped_reward    : add potential-based Manhattan shaping
    max_steps        : episode horizon
    random_start_goal: randomise start & goal each reset (helps on-policy methods)
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        n_obstacles: int = 14,
        seed: int = 42,
        dynamic_fraction: float = 0.0,
        motion_mode: str = "patrol",
        shaped_reward: bool = False,
        max_steps: int = 200,
        random_start_goal: bool = False,
    ) -> None:
        self.H, self.W = grid_size
        self.n_obstacles = n_obstacles
        self.seed = seed
        self.dynamic_fraction = dynamic_fraction
        self.motion_mode = motion_mode
        self.shaped_reward = shaped_reward
        self.max_steps = max_steps
        self.random_start_goal = random_start_goal

        self.start: Tuple[int, int] = (0, 0)
        self.goal: Tuple[int, int] = (self.H - 1, self.W - 1)

        self._rng = np.random.RandomState(seed)
        self._episode_rng = np.random.RandomState(seed + 1000)
        self._place_obstacles()

        # Runtime state — initialised in reset()
        self.agent_pos: List[int] = list(self.start)
        self.dynamic_pos: List[List[int]] = [list(p) for p in self._dyn_init]
        self.dyn_dirs: List[Tuple[int, int]] = list(self._dyn_dir_init)
        self.steps: int = 0
        self.prev_dist: float = 0.0

    # ── obstacle placement ────────────────────────────────────────────────────

    def _place_obstacles(self) -> None:
        forbidden = {self.start, self.goal}
        candidates = [
            (r, c)
            for r in range(self.H)
            for c in range(self.W)
            if (r, c) not in forbidden
        ]

        rng = self._rng
        n = min(self.n_obstacles, len(candidates))

        # Sample until BFS confirms a valid path exists
        for _ in range(300):
            idxs = rng.choice(len(candidates), n, replace=False)
            chosen = [candidates[int(i)] for i in idxs]
            if self._bfs_reachable(set(chosen)):
                break
        else:
            chosen = chosen[: max(1, n // 2)]

        n_dyn = int(len(chosen) * self.dynamic_fraction)
        perm = rng.permutation(len(chosen)).tolist()
        chosen = [chosen[i] for i in perm]

        self._dyn_init: List[Tuple[int, int]] = [
            (int(chosen[i][0]), int(chosen[i][1])) for i in range(n_dyn)
        ]
        self._static: FrozenSet[Tuple[int, int]] = frozenset(
            (int(chosen[i][0]), int(chosen[i][1])) for i in range(n_dyn, len(chosen))
        )
        self._dyn_dir_init: List[Tuple[int, int]] = [
            (int(rng.choice([-1, 1])), int(rng.choice([-1, 1])))
            for _ in range(n_dyn)
        ]

    def _bfs_reachable(self, obs_set: Set) -> bool:
        visited: Set = {self.start}
        q: deque = deque([self.start])
        while q:
            r, c = q.popleft()
            if (r, c) == self.goal:
                return True
            for dr, dc in ACTIONS:
                nb = (r + dr, c + dc)
                if (
                    0 <= nb[0] < self.H
                    and 0 <= nb[1] < self.W
                    and nb not in obs_set
                    and nb not in visited
                ):
                    visited.add(nb)
                    q.append(nb)
        return False

    # ── Gym-style API ─────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        if self.random_start_goal:
            # Pick random start and goal from free cells, ensuring they differ
            obs = self._static
            free = [(r, c) for r in range(self.H) for c in range(self.W)
                    if (r, c) not in obs]
            if len(free) >= 2:
                idxs = self._episode_rng.choice(len(free), 2, replace=False)
                self.start = free[idxs[0]]
                self.goal  = free[idxs[1]]
        self.agent_pos = list(self.start)
        self.dynamic_pos = [list(p) for p in self._dyn_init]
        self.dyn_dirs = list(self._dyn_dir_init)
        self.steps = 0
        r, c = self.agent_pos
        gr, gc = self.goal
        self.prev_dist = float(abs(gr - r) + abs(gc - c))
        return self._state_vec()

    def step(self, action: int):
        dr, dc = ACTIONS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        reward = -1.0
        collision = False

        if 0 <= nr < self.H and 0 <= nc < self.W:
            if (nr, nc) in self._obstacle_set():
                reward -= 10.0
                collision = True
            else:
                self.agent_pos = [nr, nc]
        # wall hit: stay, only the step penalty applies

        reached = (tuple(self.agent_pos) == self.goal)
        if reached:
            reward += 100.0

        if self.shaped_reward and not reached:
            r, c = self.agent_pos
            gr, gc = self.goal
            curr = float(abs(gr - r) + abs(gc - c))
            reward += 2.0 * (self.prev_dist - curr)   # +2 toward goal, -2 away
            self.prev_dist = curr

        self._move_obstacles()

        # dynamic obstacle steps onto agent's cell
        if not reached:
            dyn_set = {tuple(p) for p in self.dynamic_pos}
            if tuple(self.agent_pos) in dyn_set:
                reward -= 10.0
                collision = True

        self.steps += 1
        done = reached or (self.steps >= self.max_steps)

        info: Dict = {
            "success": reached,
            "collision": collision,
            "steps": self.steps,
            "pos": tuple(self.agent_pos),
        }
        return self._state_vec(), reward, done, info

    # ── state representation ──────────────────────────────────────────────────

    def _state_vec(self) -> np.ndarray:
        r, c = self.agent_pos
        gr, gc = self.goal
        obs = self._obstacle_set()

        nH = max(self.H - 1, 1)
        nW = max(self.W - 1, 1)

        pos = [r / nH, c / nW]
        gdir = [(gr - r) / nH, (gc - c) / nW]
        sensors = [self._cast_ray(ddr, ddc, obs) for ddr, ddc in SENSOR_DIRS]
        return np.array(pos + gdir + sensors, dtype=np.float32)

    def _cast_ray(self, dr: int, dc: int, obs: Set) -> float:
        r, c = self.agent_pos
        max_d = float(max(self.H, self.W))
        for steps in range(1, int(max_d) + 1):
            r += dr
            c += dc
            if not (0 <= r < self.H and 0 <= c < self.W):
                return steps / max_d
            if (r, c) in obs:
                return steps / max_d
        return 1.0

    def get_state_index(self) -> int:
        """Flat row-major cell index for tabular Q-learning."""
        r, c = self.agent_pos
        return r * self.W + c

    # ── dynamic obstacle movement ─────────────────────────────────────────────

    def _move_obstacles(self) -> None:
        if not self.dynamic_pos:
            return
        reserved: Set = set(self._static) | {self.start, self.goal}

        for i, pos in enumerate(self.dynamic_pos):
            r, c = pos

            if self.motion_mode == "patrol":
                dr, dc = self.dyn_dirs[i]
                nr, nc = r + dr, c + dc
                if not self._valid_dyn(nr, nc, reserved):
                    dr, dc = -dr, -dc
                    self.dyn_dirs[i] = (dr, dc)
                    nr, nc = r + dr, c + dc
                if self._valid_dyn(nr, nc, reserved):
                    self.dynamic_pos[i] = [nr, nc]

            elif self.motion_mode == "homing":
                ar, ac = self.agent_pos
                best: Tuple[int, int] = (r, c)
                best_d = abs(ar - r) + abs(ac - c)
                for dr2, dc2 in ACTIONS:
                    nr2, nc2 = r + dr2, c + dc2
                    if self._valid_dyn(nr2, nc2, reserved):
                        d = abs(ar - nr2) + abs(ac - nc2)
                        if d < best_d:
                            best_d, best = d, (nr2, nc2)
                self.dynamic_pos[i] = list(best)

            elif self.motion_mode == "random":
                opts = [
                    (r + dr2, c + dc2)
                    for dr2, dc2 in ACTIONS
                    if self._valid_dyn(r + dr2, c + dc2, reserved)
                ]
                if opts:
                    idx = int(self._rng.randint(len(opts)))
                    self.dynamic_pos[i] = list(opts[idx])

    def _valid_dyn(self, r: int, c: int, reserved: Set) -> bool:
        return (
            0 <= r < self.H
            and 0 <= c < self.W
            and (r, c) not in reserved
        )

    def _obstacle_set(self) -> Set:
        return set(self._static) | {tuple(p) for p in self.dynamic_pos}

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def n_states(self) -> int:
        return self.H * self.W

    @property
    def n_actions(self) -> int:
        return 4

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    @property
    def static_obstacles(self) -> FrozenSet:
        return self._static

    def render_text(self) -> str:
        grid = [["." for _ in range(self.W)] for _ in range(self.H)]
        for r2, c2 in self._static:
            grid[r2][c2] = "#"
        for r2, c2 in self.dynamic_pos:
            grid[r2][c2] = "O"
        sr, sc = self.start
        gr2, gc2 = self.goal
        ar, ac = self.agent_pos
        grid[sr][sc] = "S"
        grid[gr2][gc2] = "G"
        grid[ar][ac] = "A"
        return "\n".join(" ".join(row) for row in grid)

    def get_obstacle_snapshot(self) -> Dict:
        return {
            "static": sorted(self._static),
            "dynamic": [tuple(p) for p in self.dynamic_pos],
        }
