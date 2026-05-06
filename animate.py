"""
GIF animation utilities for the RL navigation project.

animate_agent_path()   — animate a single agent traversal
animate_dijkstra()     — animate Dijkstra's exploration + final path
Both functions render frames with matplotlib, then assemble them with Pillow.

Visual conventions:
  Dark (#2C3E50)   — static obstacles
  Orange (#E67E22) — dynamic obstacles
  Green  (#27AE60) — start cell
  Gold   (#F1C40F) — goal cell
  Red arrow        — agent (direction of last move)
  Coloured trail   — method-specific path so-far
"""

from __future__ import annotations

import io
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── colour constants ───────────────────────────────────────────────────────────
_PALETTE: Dict[str, str] = {
    "Q-Learning": "#E74C3C",
    "DQN":        "#3498DB",
    "PPO":        "#2ECC71",
    "Dijkstra":   "#9B59B6",
}
_C_BG      = "#F7F9FC"
_C_GRID    = "#CBD0DA"
_C_STATIC  = "#2C3E50"
_C_DYNAMIC = "#E67E22"
_C_START   = "#27AE60"
_C_GOAL    = "#F1C40F"
_C_EXPLORE = "#AED6F1"


# ── frame → PIL Image ─────────────────────────────────────────────────────────
def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def _draw_base_grid(
    ax: plt.Axes,
    H: int,
    W: int,
    static_obs: set,
    dynamic_obs: List[Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> None:
    ax.set_facecolor(_C_BG)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # cell backgrounds + grid
    for r in range(H):
        for c in range(W):
            ax.add_patch(plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor=_C_BG, edgecolor=_C_GRID, linewidth=0.5, zorder=1,
            ))

    # static obstacles
    for r, c in static_obs:
        ax.add_patch(plt.Rectangle(
            (c - 0.5, r - 0.5), 1, 1,
            facecolor=_C_STATIC, zorder=2,
        ))

    # dynamic obstacles
    for r, c in dynamic_obs:
        ax.add_patch(plt.Rectangle(
            (c - 0.5, r - 0.5), 1, 1,
            facecolor=_C_DYNAMIC, zorder=2,
        ))
        ax.text(c, r, "▲", ha="center", va="center", fontsize=7.5,
                color="white", zorder=3)

    # start
    sr, sc = start
    ax.add_patch(plt.Circle((sc, sr), 0.38, color=_C_START, zorder=3))
    ax.text(sc, sr, "S", ha="center", va="center", fontsize=9,
            fontweight="bold", color="white", zorder=4)

    # goal
    gr, gc = goal
    ax.add_patch(plt.Rectangle(
        (gc - 0.42, gr - 0.42), 0.84, 0.84,
        facecolor=_C_GOAL, zorder=3, linewidth=0,
    ))
    ax.text(gc, gr, "G", ha="center", va="center", fontsize=9,
            fontweight="bold", color="#333", zorder=4)


def _draw_agent(
    ax: plt.Axes,
    pos: Tuple[int, int],
    prev_pos: Optional[Tuple[int, int]],
    color: str,
) -> None:
    r, c = pos
    # directional arrow
    if prev_pos and prev_pos != pos:
        pr, pc = prev_pos
        dr = r - pr
        dc = c - pc
        ax.annotate(
            "", xy=(c, r), xytext=(c - dc * 0.45, r - dr * 0.45),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=2.0,
                mutation_scale=18,
            ),
            zorder=8,
        )
    ax.add_patch(plt.Circle((c, r), 0.28, color=color, zorder=7))
    ax.add_patch(plt.Circle((c, r), 0.28, color="white", fill=False, linewidth=1.5, zorder=8))


def _draw_trail(
    ax: plt.Axes,
    path_so_far: List[Tuple[int, int]],
    color: str,
) -> None:
    if len(path_so_far) < 2:
        return
    T = len(path_so_far)
    for i in range(1, T):
        pr, pc = path_so_far[i - 1]
        nr, nc = path_so_far[i]
        alpha = 0.2 + 0.65 * (i / T)
        ax.plot([pc, nc], [pr, nr], color=color, linewidth=2.0,
                alpha=alpha, solid_capstyle="round", zorder=5)


def _legend_patches(method: str, has_dynamic: bool) -> List:
    patches = [
        mpatches.Patch(color=_C_STATIC, label="Static obstacle"),
        mpatches.Patch(color=_C_START, label="Start"),
        mpatches.Patch(color=_C_GOAL, label="Goal"),
        mpatches.Patch(color=_PALETTE.get(method, "#888"), label=f"{method} path"),
    ]
    if has_dynamic:
        patches.insert(1, mpatches.Patch(color=_C_DYNAMIC, label="Dynamic obstacle"))
    return patches


# ═══════════════════════════════════════════════════════════════════════════════
# Agent path animation
# ═══════════════════════════════════════════════════════════════════════════════
def animate_agent_path(
    path: List[Tuple[int, int]],
    env,
    method: str = "Agent",
    save_path: str = "outputs/animations/agent.gif",
    fps: int = 5,
    dynamic_obstacle_history: Optional[List[List[Tuple[int, int]]]] = None,
) -> None:
    """
    Animate a single agent traversal.

    Parameters
    ----------
    path                      : sequence of (row, col) positions visited
    env                       : GridWorldEnv (for grid geometry + obstacles)
    method                    : algorithm name (for title + colour)
    save_path                 : output GIF path
    fps                       : frames per second
    dynamic_obstacle_history  : list (one per step) of dynamic obstacle positions
    """
    if not path:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    H, W = env.H, env.W
    static_obs = set(env.static_obstacles)
    color = _PALETTE.get(method, "#888")
    has_dynamic = bool(dynamic_obstacle_history)

    frames: List[Image.Image] = []
    n_steps = len(path)

    for step_idx in range(n_steps):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor(_C_BG)

        dyn_pos = (
            dynamic_obstacle_history[min(step_idx, len(dynamic_obstacle_history) - 1)]
            if has_dynamic
            else []
        )
        _draw_base_grid(ax, H, W, static_obs, dyn_pos, env.start, env.goal)
        _draw_trail(ax, path[: step_idx + 1], color)

        prev = path[step_idx - 1] if step_idx > 0 else None
        _draw_agent(ax, path[step_idx], prev, color)

        # legend
        ax.legend(
            handles=_legend_patches(method, has_dynamic),
            loc="upper right", fontsize=7.5, framealpha=0.92, edgecolor="#ccc",
        )

        # title + step counter
        success_now = (path[step_idx] == env.goal)
        status = "✓ Reached Goal!" if success_now else f"Step {step_idx}"
        ax.set_title(f"{method}  —  {status}", fontsize=11, fontweight="bold", pad=6)

        fig.tight_layout(pad=0.4)
        frames.append(_fig_to_pil(fig))
        plt.close(fig)

    _save_gif(frames, save_path, fps=fps)


# ═══════════════════════════════════════════════════════════════════════════════
# Dijkstra exploration + path animation
# ═══════════════════════════════════════════════════════════════════════════════
def animate_dijkstra(
    dijkstra_result,   # DijkstraResult
    env,
    save_path: str = "outputs/animations/dijkstra.gif",
    fps: int = 8,
) -> None:
    if not dijkstra_result.success:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    H, W = env.H, env.W
    static_obs = set(env.static_obstacles)
    path = dijkstra_result.path
    explored = dijkstra_result.explored

    # Phase 1: exploration frames (every 3rd explored cell for brevity)
    frames: List[Image.Image] = []
    explored_so_far: List[Tuple[int, int]] = []
    step_indices = list(range(0, len(explored), max(1, len(explored) // 40))) + [len(explored) - 1]

    for si in step_indices:
        explored_so_far = explored[: si + 1]
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor(_C_BG)
        _draw_base_grid(ax, H, W, static_obs, [], env.start, env.goal)

        for r, c in explored_so_far:
            if (r, c) not in static_obs and (r, c) != env.start and (r, c) != env.goal:
                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor=_C_EXPLORE, alpha=0.55, zorder=2,
                ))

        patches = [
            mpatches.Patch(color=_C_STATIC, label="Obstacle"),
            mpatches.Patch(color=_C_EXPLORE, label="Explored"),
            mpatches.Patch(color=_C_START, label="Start"),
            mpatches.Patch(color=_C_GOAL, label="Goal"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=7.5,
                  framealpha=0.92, edgecolor="#ccc")
        ax.set_title(f"Dijkstra — Exploring  ({len(explored_so_far)}/{len(explored)} cells)",
                     fontsize=10, fontweight="bold", pad=6)
        fig.tight_layout(pad=0.4)
        frames.append(_fig_to_pil(fig))
        plt.close(fig)

    # Phase 2: final path frames
    color = _PALETTE["Dijkstra"]
    for step_idx in range(len(path)):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor(_C_BG)
        _draw_base_grid(ax, H, W, static_obs, [], env.start, env.goal)

        # show full explored region
        for r, c in explored:
            if (r, c) not in static_obs and (r, c) != env.start and (r, c) != env.goal:
                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor=_C_EXPLORE, alpha=0.3, zorder=2,
                ))

        _draw_trail(ax, path[: step_idx + 1], color)
        prev = path[step_idx - 1] if step_idx > 0 else None
        _draw_agent(ax, path[step_idx], prev, color)

        status = "✓ Reached Goal!" if step_idx == len(path) - 1 else f"Step {step_idx}"
        ax.set_title(f"Dijkstra — {status}  ({dijkstra_result.steps} steps optimal)",
                     fontsize=10, fontweight="bold", pad=6)

        patches = [
            mpatches.Patch(color=_C_EXPLORE, label="Explored"),
            mpatches.Patch(color=color, label="Optimal path"),
            mpatches.Patch(color=_C_STATIC, label="Obstacle"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=7.5,
                  framealpha=0.92, edgecolor="#ccc")

        fig.tight_layout(pad=0.4)
        frames.append(_fig_to_pil(fig))
        plt.close(fig)

    _save_gif(frames, save_path, fps=fps)


# ═══════════════════════════════════════════════════════════════════════════════
# Side-by-side comparison GIF (all methods same episode)
# ═══════════════════════════════════════════════════════════════════════════════
def animate_comparison(
    paths: Dict[str, List[Tuple[int, int]]],
    env,
    save_path: str = "outputs/animations/comparison.gif",
    fps: int = 5,
    dynamic_histories: Optional[Dict[str, Optional[List[List[Tuple[int, int]]]]]] = None,
) -> None:
    """Render all method paths side-by-side in a single GIF with moving dynamic obstacles."""
    if not paths:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    methods = list(paths.keys())
    n = len(methods)
    max_steps = max(len(p) for p in paths.values())
    H, W = env.H, env.W
    static_obs = set(env.static_obstacles)
    has_dynamic = dynamic_histories and any(v for v in dynamic_histories.values())

    frames: List[Image.Image] = []

    for step_idx in range(max_steps):
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
        fig.patch.set_facecolor(_C_BG)
        if n == 1:
            axes = [axes]
        fig.suptitle("Algorithm Comparison", fontsize=13, fontweight="bold")

        for ax, method in zip(axes, methods):
            path = paths[method]
            color = _PALETTE.get(method, "#888")
            cur_idx = min(step_idx, len(path) - 1)

            # Get dynamic obstacle positions for this method at this step
            dyn_h = (dynamic_histories or {}).get(method)
            dyn_pos = []
            if dyn_h:
                dyn_pos = dyn_h[min(cur_idx, len(dyn_h) - 1)]

            _draw_base_grid(ax, H, W, static_obs, dyn_pos, env.start, env.goal)
            _draw_trail(ax, path[: cur_idx + 1], color)

            prev = path[cur_idx - 1] if cur_idx > 0 else None
            _draw_agent(ax, path[cur_idx], prev, color)

            reached = (path[cur_idx] == env.goal)
            status = f"✓ {len(path)-1} steps" if reached else f"Step {cur_idx}"
            ax.set_title(f"{method}\n{status}", fontsize=9.5, fontweight="bold")

        fig.tight_layout(pad=0.5)
        frames.append(_fig_to_pil(fig))
        plt.close(fig)

    _save_gif(frames, save_path, fps=fps)


# ── GIF writer ────────────────────────────────────────────────────────────────
def _save_gif(frames: List[Image.Image], path: str, fps: int = 5) -> None:
    if not frames:
        return
    # Ensure RGBA → RGB for compatibility
    rgb_frames = [f.convert("RGB") for f in frames]
    duration_ms = int(1000 / fps)
    rgb_frames[0].save(
        path,
        save_all=True,
        append_images=rgb_frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    print(f"  Saved GIF → {path}  ({len(rgb_frames)} frames, {fps} fps)")
