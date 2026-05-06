"""
Presentation-quality visualisations for the RL navigation project.

All plots share:
  • Consistent per-method colour palette
  • Clean spines (bottom + left only)
  • Smooth learning curves (rolling mean ± std-dev band)
  • 200 dpi PNG output — suitable for reports and slides
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9.5,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.7,
    "lines.linewidth": 2.2,
})

# ── colour palette ─────────────────────────────────────────────────────────────
PALETTE: Dict[str, str] = {
    "Q-Learning": "#E74C3C",   # scarlet
    "DQN":        "#3498DB",   # azure
    "PPO":        "#2ECC71",   # emerald
    "Dijkstra":   "#9B59B6",   # amethyst
}
LINESTYLES: Dict[str, str] = {
    "Q-Learning": "solid",
    "DQN":        "dashed",
    "PPO":        "dashdot",
    "Dijkstra":   "dotted",
}

# grid / obstacle colours
_C_BG      = "#F7F9FC"   # cell background
_C_GRID    = "#CBD0DA"   # grid lines
_C_STATIC  = "#2C3E50"   # static obstacle
_C_DYNAMIC = "#E67E22"   # dynamic obstacle
_C_START   = "#27AE60"   # start cell
_C_GOAL    = "#F1C40F"   # goal cell
_C_AGENT   = "#E74C3C"   # agent position
_C_EXPLORE = "#AED6F1"   # Dijkstra explored cell


# ── helpers ───────────────────────────────────────────────────────────────────
def _smooth(arr: np.ndarray, w: int = 30) -> np.ndarray:
    """Rolling mean with a wider default window for smoother presentation curves."""
    if len(arr) < w:
        w = max(1, len(arr) // 3)
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def _rolling_stats(arr: np.ndarray, w: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_indices, mean, std) of rolling window."""
    means, stds = [], []
    for i in range(w, len(arr) + 1):
        window = arr[i - w : i]
        means.append(np.mean(window))
        stds.append(np.std(window))
    x = np.arange(w, len(arr) + 1)
    return x, np.array(means), np.array(stds)


def _style_ax(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(_C_BG)
    if title:
        ax.set_title(title, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _legend(ax: plt.Axes, **kw) -> None:
    ax.legend(loc="best", **kw)


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Learning curves (reward per episode)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_learning_curves(
    histories: Dict[str, List[float]],    # method → list of episode rewards
    title: str = "Training Reward",
    path: str = "outputs/plots/learning_curves.png",
    window: int = 20,
    exp_label: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style_ax(ax, title=f"{title}{' — ' + exp_label if exp_label else ''}", xlabel="Episode", ylabel="Episode Reward")

    for method, rewards in histories.items():
        if not rewards:
            continue
        arr = np.array(rewards, dtype=float)
        x, mu, sd = _rolling_stats(arr, window)
        c = PALETTE.get(method, "#666")
        ls = LINESTYLES.get(method, "solid")
        ax.plot(x, mu, color=c, linestyle=ls, label=method, zorder=3)
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15, zorder=2)

    _legend(ax)
    fig.tight_layout()
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Success-rate curves
# ═══════════════════════════════════════════════════════════════════════════════
def plot_success_curves(
    histories: Dict[str, List[bool]],
    title: str = "Success Rate",
    path: str = "outputs/plots/success_curves.png",
    window: int = 20,
    exp_label: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style_ax(ax, title=f"{title}{' — ' + exp_label if exp_label else ''}", xlabel="Episode", ylabel="Success Rate")

    for method, successes in histories.items():
        if not successes:
            continue
        arr = np.array(successes, dtype=float)
        x, mu, _ = _rolling_stats(arr, window)
        c = PALETTE.get(method, "#666")
        ls = LINESTYLES.get(method, "solid")
        ax.plot(x, mu, color=c, linestyle=ls, label=method, zorder=3)

    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    _legend(ax)
    fig.tight_layout()
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Steps-per-episode curves
# ═══════════════════════════════════════════════════════════════════════════════
def plot_steps_curves(
    histories: Dict[str, List[int]],
    title: str = "Steps per Episode",
    path: str = "outputs/plots/steps_curves.png",
    window: int = 20,
    exp_label: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style_ax(ax, title=f"{title}{' — ' + exp_label if exp_label else ''}", xlabel="Episode", ylabel="Steps")

    for method, steps in histories.items():
        if not steps:
            continue
        arr = np.array(steps, dtype=float)
        x, mu, sd = _rolling_stats(arr, window)
        c = PALETTE.get(method, "#666")
        ls = LINESTYLES.get(method, "solid")
        ax.plot(x, mu, color=c, linestyle=ls, label=method, zorder=3)
        ax.fill_between(x, np.clip(mu - sd, 0, None), mu + sd, color=c, alpha=0.12, zorder=2)

    _legend(ax)
    fig.tight_layout()
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DQN loss curve
# ═══════════════════════════════════════════════════════════════════════════════
def plot_loss_curve(
    losses: List[float],
    path: str = "outputs/plots/dqn_loss.png",
    window: int = 20,
) -> None:
    arr = np.array([l for l in losses if l > 0], dtype=float)
    if len(arr) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    _style_ax(ax, title="DQN Training Loss (Huber)", xlabel="Update Step", ylabel="Loss")
    x = np.arange(len(arr))
    ax.plot(x, arr, color=PALETTE["DQN"], alpha=0.3, linewidth=1)
    if len(arr) >= window:
        sm = _smooth(arr, window)
        ax.plot(np.arange(window - 1, len(arr)), sm, color=PALETTE["DQN"], linewidth=2, label=f"{window}-ep MA")
        _legend(ax)
    fig.tight_layout()
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Method comparison bar chart (eval metrics)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_method_comparison(
    eval_results: List[Any],   # List[EvalResult]
    title: str = "Method Comparison",
    path: str = "outputs/plots/method_comparison.png",
    exp_label: str = "",
) -> None:
    methods = [r.method for r in eval_results]
    metrics = {
        "Success Rate (%)": [r.success_rate * 100 for r in eval_results],
        "Avg Reward":        [r.avg_reward for r in eval_results],
        "Avg Steps":         [r.avg_steps for r in eval_results],
        "Avg Path Length":   [r.avg_path_length for r in eval_results],
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    fig.suptitle(f"{title}{' — ' + exp_label if exp_label else ''}", y=1.02)

    for ax, (metric, vals) in zip(axes, metrics.items()):
        colors = [PALETTE.get(m, "#aaa") for m in methods]
        bars = ax.bar(methods, vals, color=colors, edgecolor="white", linewidth=1.2, width=0.55)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(bar.get_height()) * 0.02,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="semibold",
            )
        _style_ax(ax, title=metric)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)

    fig.tight_layout()
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Aggregated multi-experiment comparison (with error bars across experiments)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_aggregate_comparison(
    exp_results: List[List[Any]],   # [exp1_results, exp2_results, ...]
    exp_labels: List[str],
    path: str = "outputs/plots/aggregate_comparison.png",
) -> None:
    """Bar chart grouped by algorithm; each group has one bar per experiment."""
    all_methods = [r.method for r in exp_results[0]]
    n_methods = len(all_methods)
    n_exps = len(exp_results)

    metrics = ["success_rate", "avg_reward", "avg_steps", "avg_path_length"]
    metric_labels = ["Success Rate (%)", "Avg Reward", "Avg Steps", "Avg Path Length"]
    scale = [100, 1, 1, 1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Performance Across All Experiments", y=1.02)

    bar_w = 0.22
    x = np.arange(n_methods)
    cmap = ["#3498DB", "#E67E22", "#8E44AD"]  # one shade per experiment

    for ax, (met, label, sc) in zip(axes, zip(metrics, metric_labels, scale)):
        for ei, (exp_r, elabel) in enumerate(zip(exp_results, exp_labels)):
            vals = [getattr(r, met) * sc for r in exp_r]
            offset = (ei - (n_exps - 1) / 2) * bar_w
            bars = ax.bar(
                x + offset, vals, width=bar_w,
                color=cmap[ei], alpha=0.85, edgecolor="white",
                label=elabel, linewidth=0.8,
            )
        _style_ax(ax, title=label)
        ax.set_xticks(x)
        ax.set_xticklabels(all_methods, rotation=20, ha="right", fontsize=8.5)
        if ax is axes[0]:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Policy heatmap — Q-Learning
# ═══════════════════════════════════════════════════════════════════════════════
def plot_q_policy(
    agent,
    env,
    path: str = "outputs/plots/q_policy.png",
    exp_label: str = "",
) -> None:
    H, W = env.H, env.W
    max_q = np.max(agent.Q, axis=1).reshape(H, W)
    actions = np.argmax(agent.Q, axis=1).reshape(H, W)

    arrow_map = {0: (0, -0.35), 1: (0, 0.35), 2: (-0.35, 0), 3: (0.35, 0)}  # U/D/L/R

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(
        max_q, cmap="RdYlGn", origin="upper",
        vmin=np.percentile(max_q, 5), vmax=np.percentile(max_q, 95),
    )
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="max Q-value")

    obs = set(env.static_obstacles)
    for r in range(H):
        for c in range(W):
            cell = (r, c)
            if cell in obs:
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=_C_STATIC, zorder=3)
                ax.add_patch(rect)
            elif cell == env.goal:
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=_C_GOAL, zorder=3)
                ax.add_patch(rect)
                ax.text(c, r, "G", ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)
            elif cell == env.start:
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=_C_START, alpha=0.7, zorder=3)
                ax.add_patch(rect)
                ax.text(c, r, "S", ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)
            else:
                dx, dy = arrow_map[actions[r, c]]
                ax.annotate(
                    "", xy=(c + dx, r + dy), xytext=(c, r),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
                    zorder=4,
                )

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.set_xticklabels(range(W), fontsize=8); ax.set_yticklabels(range(H), fontsize=8)
    ax.set_title(f"Q-Learning Policy{' — ' + exp_label if exp_label else ''}", pad=10)
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    fig.tight_layout()
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Policy heatmap — DQN
# ═══════════════════════════════════════════════════════════════════════════════
def plot_dqn_policy(
    agent,
    env,
    path: str = "outputs/plots/dqn_policy.png",
    exp_label: str = "",
) -> None:
    import torch
    H, W = env.H, env.W
    arrow_map = {0: (0, -0.35), 1: (0, 0.35), 2: (-0.35, 0), 3: (0.35, 0)}

    q_grid = np.zeros((H, W, 4))
    with torch.no_grad():
        for r in range(H):
            for c in range(W):
                env.agent_pos = [r, c]
                sv = env._state_vec()
                x = torch.from_numpy(sv).unsqueeze(0).to(agent.device)
                q_grid[r, c] = agent.online(x).cpu().numpy()[0]

    max_q = q_grid.max(axis=2)
    actions = q_grid.argmax(axis=2)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(
        max_q, cmap="RdYlGn", origin="upper",
        vmin=np.percentile(max_q, 5), vmax=np.percentile(max_q, 95),
    )
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="max Q-value")

    obs = set(env.static_obstacles)
    for r in range(H):
        for c in range(W):
            cell = (r, c)
            if cell in obs:
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=_C_STATIC, zorder=3)
                ax.add_patch(rect)
            elif cell == env.goal:
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=_C_GOAL, zorder=3)
                ax.add_patch(rect)
                ax.text(c, r, "G", ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)
            elif cell == env.start:
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=_C_START, alpha=0.7, zorder=3)
                ax.add_patch(rect)
                ax.text(c, r, "S", ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)
            else:
                dx, dy = arrow_map[actions[r, c]]
                ax.annotate(
                    "", xy=(c + dx, r + dy), xytext=(c, r),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
                    zorder=4,
                )

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.set_xticklabels(range(W), fontsize=8); ax.set_yticklabels(range(H), fontsize=8)
    ax.set_title(f"DQN Policy{' — ' + exp_label if exp_label else ''}", pad=10)
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    fig.tight_layout()
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Path comparison (all methods on the same grid)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_path_comparison(
    paths: Dict[str, List[Tuple[int, int]]],   # method → path list
    env,
    dijkstra_explored: Optional[List[Tuple[int, int]]] = None,
    title: str = "Path Comparison",
    path: str = "outputs/plots/path_comparison.png",
    exp_label: str = "",
) -> None:
    H, W = env.H, env.W
    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(f"{title}{' — ' + exp_label if exp_label else ''}", y=1.02)

    for ax, (method, agent_path) in zip(axes, paths.items()):
        _draw_grid(ax, env, title=method)

        # For Dijkstra: show explored region first
        if method == "Dijkstra" and dijkstra_explored:
            for cell in dijkstra_explored:
                r, c = cell
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                     color=_C_EXPLORE, alpha=0.4, zorder=2)
                ax.add_patch(rect)

        # Draw path
        if agent_path:
            rs = [p[0] for p in agent_path]
            cs = [p[1] for p in agent_path]
            c_col = PALETTE.get(method, "#888")
            ax.plot(cs, rs, color=c_col, linewidth=2.5, zorder=5, alpha=0.85,
                    solid_capstyle="round", solid_joinstyle="round")
            # step markers
            ax.scatter(cs[1:-1], rs[1:-1], s=18, color=c_col, zorder=6, alpha=0.6)

        steps_lbl = f"{len(agent_path) - 1} steps" if agent_path else "no path"
        ax.set_title(f"{method}\n({steps_lbl})", fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Parameter sweep plots
# ═══════════════════════════════════════════════════════════════════════════════
def plot_parameter_sweep(
    sweep_results: Dict[str, Dict[Any, List[float]]],
    param_name: str,
    metric_name: str = "Success Rate",
    title: str = "",
    path: str = "outputs/plots/param_sweep.png",
) -> None:
    """
    sweep_results: { method: { param_value: list_of_metric_values_per_seed } }
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _style_ax(ax,
              title=title or f"Parameter Sweep — {param_name}",
              xlabel=param_name,
              ylabel=metric_name)

    for method, val_dict in sweep_results.items():
        param_vals = sorted(val_dict.keys())
        means = [np.mean(val_dict[v]) for v in param_vals]
        stds  = [np.std(val_dict[v])  for v in param_vals]
        c = PALETTE.get(method, "#aaa")
        ax.plot(param_vals, means, "o-", color=c, label=method, markersize=6, zorder=3)
        ax.fill_between(
            param_vals,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=c, alpha=0.15, zorder=2,
        )

    _legend(ax)
    fig.tight_layout()
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Reward shaping comparison
# ═══════════════════════════════════════════════════════════════════════════════
def plot_reward_shaping(
    shaped: Dict[str, List[float]],
    unshaped: Dict[str, List[float]],
    path: str = "outputs/plots/reward_shaping.png",
    window: int = 20,
) -> None:
    methods = [m for m in shaped if m in unshaped]
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Effect of Reward Shaping on Convergence", y=1.02)

    for ax, method in zip(axes, methods):
        _style_ax(ax, title=method, xlabel="Episode", ylabel="Reward (rolling mean)")
        for label, data, ls in [("Shaped", shaped[method], "solid"),
                                 ("Unshaped", unshaped[method], "dashed")]:
            arr = np.array(data, dtype=float)
            x, mu, sd = _rolling_stats(arr, window)
            c = PALETTE.get(method, "#888")
            ax.plot(x, mu, color=c, linestyle=ls, label=label, zorder=3)
            ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.12)
        _legend(ax)

    fig.tight_layout()
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Summary dashboard (single figure, 6 panels)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_summary_dashboard(
    histories: Dict[str, "TrainingHistory"],
    eval_results: List[Any],
    env,
    path: str = "outputs/plots/summary_dashboard.png",
    exp_label: str = "",
) -> None:
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    title_str = f"RL Navigation — Summary Dashboard{' (' + exp_label + ')' if exp_label else ''}"
    fig.suptitle(title_str, fontsize=16, fontweight="bold")

    # Panel 1: reward curves
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, title="Training Reward", xlabel="Episode", ylabel="Reward")
    for method, h in histories.items():
        if not h.rewards:
            continue
        arr = np.array(h.rewards, dtype=float)
        x, mu, sd = _rolling_stats(arr, 20)
        c = PALETTE.get(method, "#888")
        ax1.plot(x, mu, color=c, label=method)
        ax1.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.12)
    ax1.legend(fontsize=8)

    # Panel 2: success curves
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, title="Success Rate", xlabel="Episode", ylabel="Success Rate")
    for method, h in histories.items():
        if not h.successes:
            continue
        arr = np.array(h.successes, dtype=float)
        x, mu, _ = _rolling_stats(arr, 20)
        ax2.plot(x, mu, color=PALETTE.get(method, "#888"), label=method)
    ax2.set_ylim(-0.02, 1.05)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.legend(fontsize=8)

    # Panel 3: steps curves
    ax3 = fig.add_subplot(gs[0, 2])
    _style_ax(ax3, title="Steps per Episode", xlabel="Episode", ylabel="Steps")
    for method, h in histories.items():
        if not h.steps:
            continue
        arr = np.array(h.steps, dtype=float)
        x, mu, sd = _rolling_stats(arr, 20)
        ax3.plot(x, mu, color=PALETTE.get(method, "#888"), label=method)
    ax3.legend(fontsize=8)

    # Panel 4: success bar
    ax4 = fig.add_subplot(gs[1, 0])
    _style_ax(ax4, title="Eval Success Rate")
    methods = [r.method for r in eval_results]
    vals = [r.success_rate * 100 for r in eval_results]
    colors = [PALETTE.get(m, "#aaa") for m in methods]
    bars = ax4.bar(methods, vals, color=colors, edgecolor="white", linewidth=1)
    for bar, v in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax4.set_ylim(0, 115)
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels(methods, rotation=15, ha="right")

    # Panel 5: avg reward bar
    ax5 = fig.add_subplot(gs[1, 1])
    _style_ax(ax5, title="Eval Avg Reward")
    vals5 = [r.avg_reward for r in eval_results]
    bars = ax5.bar(methods, vals5, color=colors, edgecolor="white", linewidth=1)
    for bar, v in zip(bars, vals5):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + abs(bar.get_height()) * 0.02,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax5.set_xticks(range(len(methods)))
    ax5.set_xticklabels(methods, rotation=15, ha="right")

    # Panel 6: avg steps bar
    ax6 = fig.add_subplot(gs[1, 2])
    _style_ax(ax6, title="Eval Avg Steps")
    vals6 = [r.avg_steps for r in eval_results]
    bars = ax6.bar(methods, vals6, color=colors, edgecolor="white", linewidth=1)
    for bar, v in zip(bars, vals6):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax6.set_xticks(range(len(methods)))
    ax6.set_xticklabels(methods, rotation=15, ha="right")

    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Convergence comparison bar chart
# ═══════════════════════════════════════════════════════════════════════════════
def plot_convergence_bar(
    histories: Dict[str, Any],
    path: str = "outputs/plots/convergence.png",
    exp_label: str = "",
) -> None:
    methods, conv_eps = [], []
    for method, h in histories.items():
        ep = h.convergence_episode()
        methods.append(method)
        conv_eps.append(ep if ep is not None else len(h.rewards))

    colors = [PALETTE.get(m, "#aaa") for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4))
    _style_ax(ax, title=f"Convergence Episode (90% success){' — ' + exp_label if exp_label else ''}",
              xlabel="Method", ylabel="Episode")
    bars = ax.bar(methods, conv_eps, color=colors, edgecolor="white", linewidth=1.2, width=0.55)
    for bar, ep in zip(bars, conv_eps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                str(ep), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha="right")
    fig.tight_layout()
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# helpers used by several plot functions
# ═══════════════════════════════════════════════════════════════════════════════
def _draw_grid(ax: plt.Axes, env, title: str = "") -> None:
    H, W = env.H, env.W
    ax.set_facecolor(_C_BG)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.set_xticklabels(range(W), fontsize=7); ax.set_yticklabels(range(H), fontsize=7)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")

    # cell backgrounds
    for r in range(H):
        for c in range(W):
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor=_C_BG, edgecolor=_C_GRID, linewidth=0.6, zorder=1))

    # static obstacles
    for r, c in env.static_obstacles:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   facecolor=_C_STATIC, zorder=2))

    # dynamic obstacles
    for r, c in env.dynamic_pos:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   facecolor=_C_DYNAMIC, zorder=2))

    # start / goal
    sr, sc = env.start
    gr, gc = env.goal
    ax.add_patch(plt.Circle((sc, sr), 0.38, color=_C_START, zorder=3))
    ax.text(sc, sr, "S", ha="center", va="center", fontsize=8, fontweight="bold",
            color="white", zorder=4)
    ax.add_patch(plt.Rectangle((gc - 0.42, gr - 0.42), 0.84, 0.84,
                                facecolor=_C_GOAL, zorder=3, linewidth=0))
    ax.text(gc, gr, "G", ha="center", va="center", fontsize=8, fontweight="bold",
            color="#333", zorder=4)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Difficulty progression — success rate across experiments
# ═══════════════════════════════════════════════════════════════════════════════
def plot_difficulty_progression(
    all_exp_results: List[Dict],   # [{label, tag, eval_results:[EvalResult,...]}]
    path: str = "outputs/plots/difficulty_progression.png",
) -> None:
    """
    Grouped bar chart showing how each RL method degrades as obstacle
    difficulty increases (static → mild dynamic → heavy dynamic).
    """
    methods = ["Q-Learning", "DQN", "PPO"]
    labels  = [d["label"] for d in all_exp_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.patch.set_facecolor(_C_BG)
    fig.suptitle("Performance Across Increasing Difficulty", fontsize=14, fontweight="bold")

    metrics = [
        ("success_rate",    "Success Rate",    "% Episodes"),
        ("avg_reward",      "Avg Reward",      "Reward"),
        ("avg_path_length", "Avg Path Length", "Steps"),
    ]

    for ax, (field, title, ylabel) in zip(axes, metrics):
        ax.set_facecolor(_C_BG)
        x = np.arange(len(labels))
        width = 0.22
        offsets = np.linspace(-width, width, len(methods))

        for offset, method in zip(offsets, methods):
            vals = []
            for exp_d in all_exp_results:
                er = next((r for r in exp_d["eval_results"] if r.method == method), None)
                v = getattr(er, field, 0) if er else 0
                # Success rate as percentage
                if field == "success_rate":
                    v = v * 100
                vals.append(v)
            color = PALETTE.get(method, "#888")
            bars = ax.bar(x + offset, vals, width=width * 0.9,
                          color=color, edgecolor="white", linewidth=0.8, label=method)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (0.01 if field == "success_rate" else 0.5),
                            f"{v:.0f}{'%' if field=='success_rate' else ''}",
                            ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        if ax == axes[0]:
            ax.legend(fontsize=8.5, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, path)
