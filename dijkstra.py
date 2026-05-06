"""
Dijkstra's algorithm for shortest-path planning on a grid.

With uniform step costs this is equivalent to BFS but uses a priority queue
(min-heap) as required by the classical Dijkstra formulation.

Returns the optimal path, exploration order, total cost and step count so the
caller can both evaluate path quality and visualise the search process.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass
class DijkstraResult:
    path: List[Tuple[int, int]]      # sequence of cells from start to goal
    explored: List[Tuple[int, int]]  # cells popped from the priority queue (exploration order)
    cost: float                       # total path cost
    steps: int                        # number of moves (= len(path) - 1)
    success: bool

    # convenience: reward mimicking the RL environment
    @property
    def reward(self) -> float:
        if not self.success:
            return -float(self.steps)
        return 100.0 - float(self.steps)


# ── actions (4-connected grid) ────────────────────────────────────────────────
_ACTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def run_dijkstra(
    grid_size: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> DijkstraResult:
    """
    Classic Dijkstra on an H×W grid with unit edge costs.

    Parameters
    ----------
    grid_size : (H, W)
    obstacles : set of impassable cells
    start     : (row, col) source
    goal      : (row, col) destination
    """
    H, W = grid_size

    dist: Dict[Tuple[int, int], float] = {start: 0.0}
    prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    explored: List[Tuple[int, int]] = []

    # heap entries: (cost, row, col)
    heap: List[Tuple[float, int, int]] = [(0.0, start[0], start[1])]

    while heap:
        cost, r, c = heapq.heappop(heap)
        node = (r, c)

        if cost > dist.get(node, float("inf")):
            continue  # stale entry

        explored.append(node)

        if node == goal:
            path = _reconstruct(prev, goal)
            return DijkstraResult(
                path=path,
                explored=explored,
                cost=cost,
                steps=len(path) - 1,
                success=True,
            )

        for dr, dc in _ACTIONS:
            nb = (r + dr, c + dc)
            if not (0 <= nb[0] < H and 0 <= nb[1] < W):
                continue
            if nb in obstacles:
                continue
            new_cost = cost + 1.0
            if new_cost < dist.get(nb, float("inf")):
                dist[nb] = new_cost
                prev[nb] = node
                heapq.heappush(heap, (new_cost, nb[0], nb[1]))

    # goal unreachable
    return DijkstraResult(path=[], explored=explored, cost=float("inf"), steps=0, success=False)


def _reconstruct(
    prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
    goal: Tuple[int, int],
) -> List[Tuple[int, int]]:
    path = []
    node: Optional[Tuple[int, int]] = goal
    while node is not None:
        path.append(node)
        node = prev[node]
    return list(reversed(path))
