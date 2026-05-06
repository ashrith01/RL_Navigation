"""
main.py — Full experiment pipeline for RL robot navigation.

Experiments
-----------
  Exp 1 (seed=42,  static)          : baseline static obstacle layout
  Exp 2 (seed=189, 30% dynamic patrol): moderate dynamic obstacles

For each experiment:
  • Train Q-Learning, DQN, PPO  (500 episodes each)
  • Run Dijkstra (optimal on static snapshot)
  • Evaluate all agents (30 deterministic rollouts)
  • Save per-experiment learning curves, path plots, policy maps, GIFs

Aggregated outputs:
  • Multi-experiment comparison bar chart
  • Summary dashboard
  • Parameter sweeps (α, γ, ε_decay, lr, n_obstacles)
  • Reward shaping comparison
  • Side-by-side comparison GIF
  • CSV / JSON result tables
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# ── local modules ──────────────────────────────────────────────────────────────
from environment import GridWorldEnv
from train import train_q_learning, train_dqn, train_ppo, TrainingHistory
from evaluate import (
    evaluate_agent,
    evaluate_dijkstra,
    dijkstra_explored,
    write_eval_csv,
    write_json,
    EvalResult,
)
from visualize import (
    plot_learning_curves,
    plot_success_curves,
    plot_steps_curves,
    plot_loss_curve,
    plot_method_comparison,
    plot_aggregate_comparison,
    plot_q_policy,
    plot_dqn_policy,
    plot_path_comparison,
    plot_parameter_sweep,
    plot_reward_shaping,
    plot_summary_dashboard,
    plot_convergence_bar,
    plot_difficulty_progression,
)
from animate import animate_agent_path, animate_dijkstra, animate_comparison

# ── output directories ─────────────────────────────────────────────────────────
OUT_PLOTS = "outputs/plots"
OUT_ANIM  = "outputs/animations"
OUT_TABS  = "outputs/tables"

for d in [OUT_PLOTS, OUT_ANIM, OUT_TABS]:
    os.makedirs(d, exist_ok=True)


# ── experiment configurations ──────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "label":    "Static",
        "tag":      "exp1",
        "seed":     42,
        "dynamic_fraction": 0.0,
        "motion_mode":      "patrol",
    },
    {
        "label":    "Mild Dynamic (30% patrol)",
        "tag":      "exp2",
        "seed":     189,
        "dynamic_fraction": 0.3,
        "motion_mode":      "patrol",
    },
]

# ── training hyper-parameters ──────────────────────────────────────────────────
N_EPISODES   = 500
N_OBSTACLES  = 14
GRID_SIZE    = (10, 10)
N_EVAL_EPS   = 30
EVAL_SEED    = 99


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _make_env(cfg: Dict, shaped: bool = False, random_start_goal: bool = False) -> GridWorldEnv:
    return GridWorldEnv(
        grid_size=GRID_SIZE,
        n_obstacles=N_OBSTACLES,
        seed=cfg["seed"],
        dynamic_fraction=cfg["dynamic_fraction"],
        motion_mode=cfg["motion_mode"],
        shaped_reward=shaped,
        random_start_goal=random_start_goal,
    )


def _pp(label: str) -> None:
    print(f"\n{'='*60}\n  {label}\n{'='*60}")


def _bar(method: str, elapsed: float) -> None:
    print(f"    [{method:<12}]  done in {elapsed:5.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# per-experiment training + evaluation + plots
# ═══════════════════════════════════════════════════════════════════════════════
def run_experiment(cfg: Dict) -> Tuple[Dict[str, TrainingHistory], List[EvalResult]]:
    tag   = cfg["tag"]
    label = cfg["label"]
    _pp(f"Experiment — {label}  [{tag}]")

    env = _make_env(cfg)

    # ── training ──────────────────────────────────────────────────────────────
    t0 = time.time()
    ql_hist = train_q_learning(env, n_episodes=N_EPISODES, seed=0)
    _bar("Q-Learning", time.time() - t0)

    t0 = time.time()
    dqn_hist = train_dqn(env, n_episodes=N_EPISODES, seed=0)
    _bar("DQN", time.time() - t0)

    t0 = time.time()
    # PPO uses shaped reward for denser learning signal; conservative hyperparams prevent collapse
    ppo_env = _make_env(cfg, shaped=True)
    ppo_hist = train_ppo(ppo_env, n_episodes=N_EPISODES, seed=0)
    _bar("PPO", time.time() - t0)

    histories: Dict[str, TrainingHistory] = {
        "Q-Learning": ql_hist,
        "DQN":        dqn_hist,
        "PPO":        ppo_hist,
    }

    # ── evaluation ────────────────────────────────────────────────────────────
    ql_eval  = evaluate_agent("Q-Learning", ql_hist.agent,  env, N_EVAL_EPS, EVAL_SEED)
    dqn_eval = evaluate_agent("DQN",        dqn_hist.agent, env, N_EVAL_EPS, EVAL_SEED)
    ppo_eval = evaluate_agent("PPO",        ppo_hist.agent, env, N_EVAL_EPS, EVAL_SEED)
    dij_eval = evaluate_dijkstra(env)
    eval_results: List[EvalResult] = [ql_eval, dqn_eval, ppo_eval, dij_eval]

    for r in eval_results:
        print(f"    {r.method:<14}  success={r.success_rate:.2f}  "
              f"reward={r.avg_reward:7.2f}  steps={r.avg_steps:6.1f}")

    # ── per-experiment plots ───────────────────────────────────────────────────
    pfx = f"{OUT_PLOTS}/{tag}"

    plot_learning_curves(
        {m: h.rewards for m, h in histories.items()},
        title="Training Reward", exp_label=label,
        path=f"{pfx}_learning_curves.png",
    )
    plot_success_curves(
        {m: h.successes for m, h in histories.items()},
        title="Success Rate", exp_label=label,
        path=f"{pfx}_success_curves.png",
    )
    plot_steps_curves(
        {m: h.steps for m, h in histories.items()},
        title="Steps per Episode", exp_label=label,
        path=f"{pfx}_steps_curves.png",
    )
    plot_loss_curve(dqn_hist.losses, path=f"{pfx}_dqn_loss.png")

    plot_method_comparison(eval_results, exp_label=label,
                           path=f"{pfx}_method_comparison.png")
    plot_convergence_bar(histories, exp_label=label,
                         path=f"{pfx}_convergence.png")

    # policy maps (static env only or first static snapshot)
    if cfg["dynamic_fraction"] == 0.0:
        plot_q_policy(ql_hist.agent, env, path=f"{pfx}_q_policy.png", exp_label=label)
        plot_dqn_policy(dqn_hist.agent, env, path=f"{pfx}_dqn_policy.png", exp_label=label)

    # path comparison
    dij_res = dijkstra_explored(env)
    best_paths = {
        "Q-Learning": ql_eval.paths[0] if ql_eval.paths else [],
        "DQN":        dqn_eval.paths[0] if dqn_eval.paths else [],
        "PPO":        ppo_eval.paths[0] if ppo_eval.paths else [],
        "Dijkstra":   dij_res.path,
    }
    plot_path_comparison(
        best_paths, env,
        dijkstra_explored=dij_res.explored,
        exp_label=label,
        path=f"{pfx}_path_comparison.png",
    )

    plot_summary_dashboard(
        histories, eval_results, env,
        path=f"{pfx}_summary_dashboard.png",
        exp_label=label,
    )

    # ── GIF animations ────────────────────────────────────────────────────────
    print("  Generating GIFs …")
    from evaluate import rollout_q, rollout_dqn, rollout_ppo

    _rollout_fns = {"Q-Learning": rollout_q, "DQN": rollout_dqn, "PPO": rollout_ppo}

    def _is_stuck(path, success_rate):
        """True when greedy episode fails and the agent barely explores."""
        if not path:
            return True
        # Failed episode with fewer than 15% of free cells visited → boring GIF
        free_cells = env.H * env.W - len(env._static)
        explored_frac = len(set(path)) / max(free_cells, 1)
        return success_rate == 0.0 and explored_frac < 0.15

    def _vis_path_and_dyn(method, eval_r, agent_obj):
        """Return (path, dyn_history) for animation; use stochastic if greedy fails/is stuck."""
        path = eval_r.paths[0] if eval_r.paths else []
        dyn  = eval_r.dynamic_histories[0] if eval_r.dynamic_histories else None
        if _is_stuck(path, eval_r.success_rate):
            # Re-run stochastically so the GIF shows dynamic exploration
            p, _, _, _, _, d = _rollout_fns[method](agent_obj, env, stochastic=True)
            return p, d
        return path, dyn

    gif_paths = {}   # collect paths for comparison GIF
    gif_dyns  = {}   # collect dynamic histories (stochastic if stuck) for comparison GIF
    for method, eval_r, hist_agent in [
        ("Q-Learning", ql_eval,  ql_hist.agent),
        ("DQN",        dqn_eval, dqn_hist.agent),
        ("PPO",        ppo_eval, ppo_hist.agent),
    ]:
        vis_path, vis_dyn = _vis_path_and_dyn(method, eval_r, hist_agent)
        gif_paths[method] = vis_path
        gif_dyns[method]  = vis_dyn   # keep consistent with individual GIF
        animate_agent_path(
            path=vis_path,
            env=env,
            method=method,
            save_path=f"{OUT_ANIM}/{tag}_{method.lower().replace('-','_').replace(' ','_')}.gif",
            fps=5,
            dynamic_obstacle_history=vis_dyn,
        )

    animate_dijkstra(dij_res, env, save_path=f"{OUT_ANIM}/{tag}_dijkstra.gif", fps=8)

    # Comparison GIF — use the same paths + dynamic histories as individual GIFs
    gif_paths["Dijkstra"] = dij_res.path
    gif_dyns["Dijkstra"]  = None
    animate_comparison(
        gif_paths, env,
        save_path=f"{OUT_ANIM}/{tag}_comparison.gif",
        fps=4,
        dynamic_histories=gif_dyns,
    )

    return histories, eval_results


# ═══════════════════════════════════════════════════════════════════════════════
# parameter sweeps
# ═══════════════════════════════════════════════════════════════════════════════
def run_parameter_sweeps() -> None:
    _pp("Parameter Sweeps")

    # Use static env + 2 seeds per configuration for speed
    sweep_seeds = [0, 1]
    N_SW = 200          # episodes for each sweep run
    base_cfg = EXPERIMENTS[0]   # static env

    # ── helper ────────────────────────────────────────────────────────────────
    def _sweep_metric(hist: TrainingHistory) -> float:
        return hist.success_rate(last_n=30)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Q-Learning learning rate (α)
    # ─────────────────────────────────────────────────────────────────────────
    print("  Sweeping Q-Learning α …")
    alphas = [0.01, 0.05, 0.1, 0.3, 0.5]
    q_alpha_res: Dict[float, List[float]] = {a: [] for a in alphas}
    for a in alphas:
        for s in sweep_seeds:
            env = _make_env(base_cfg)
            h = train_q_learning(env, n_episodes=N_SW, alpha=a, seed=s)
            q_alpha_res[a].append(_sweep_metric(h))

    plot_parameter_sweep(
        {"Q-Learning": q_alpha_res},
        param_name="Learning Rate (α)",
        metric_name="Success Rate",
        title="Q-Learning — Learning Rate Sweep",
        path=f"{OUT_PLOTS}/sweep_ql_alpha.png",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Discount factor (γ) — Q-Learning & DQN
    # ─────────────────────────────────────────────────────────────────────────
    print("  Sweeping γ …")
    gammas = [0.80, 0.90, 0.95, 0.99]
    gamma_res: Dict[str, Dict] = {
        "Q-Learning": {g: [] for g in gammas},
        "DQN":        {g: [] for g in gammas},
    }
    for g in gammas:
        for s in sweep_seeds:
            env = _make_env(base_cfg)
            gamma_res["Q-Learning"][g].append(
                _sweep_metric(train_q_learning(env, n_episodes=N_SW, gamma=g, seed=s))
            )
            env = _make_env(base_cfg)
            gamma_res["DQN"][g].append(
                _sweep_metric(train_dqn(env, n_episodes=N_SW, gamma=g, seed=s))
            )

    plot_parameter_sweep(
        gamma_res,
        param_name="Discount Factor (γ)",
        metric_name="Success Rate",
        title="Discount Factor Sweep (Q-Learning vs DQN)",
        path=f"{OUT_PLOTS}/sweep_gamma.png",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. ε-decay — Q-Learning & DQN
    # ─────────────────────────────────────────────────────────────────────────
    print("  Sweeping ε-decay …")
    decays = [0.980, 0.990, 0.995, 0.999]
    decay_res: Dict[str, Dict] = {
        "Q-Learning": {d: [] for d in decays},
        "DQN":        {d: [] for d in decays},
    }
    for d in decays:
        for s in sweep_seeds:
            env = _make_env(base_cfg)
            decay_res["Q-Learning"][d].append(
                _sweep_metric(train_q_learning(env, n_episodes=N_SW, epsilon_decay=d, seed=s))
            )
            env = _make_env(base_cfg)
            decay_res["DQN"][d].append(
                _sweep_metric(train_dqn(env, n_episodes=N_SW, epsilon_decay=d, seed=s))
            )

    plot_parameter_sweep(
        decay_res,
        param_name="ε-decay rate",
        metric_name="Success Rate",
        title="Exploration Decay Sweep (Q-Learning vs DQN)",
        path=f"{OUT_PLOTS}/sweep_epsilon_decay.png",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. DQN learning rate
    # ─────────────────────────────────────────────────────────────────────────
    print("  Sweeping DQN lr …")
    lrs = [1e-4, 5e-4, 1e-3, 5e-3]
    lr_res: Dict[str, Dict] = {"DQN": {l: [] for l in lrs}}
    for l in lrs:
        for s in sweep_seeds:
            env = _make_env(base_cfg)
            lr_res["DQN"][l].append(
                _sweep_metric(train_dqn(env, n_episodes=N_SW, lr=l, seed=s))
            )

    plot_parameter_sweep(
        lr_res,
        param_name="Learning Rate (lr)",
        metric_name="Success Rate",
        title="DQN — Learning Rate Sweep",
        path=f"{OUT_PLOTS}/sweep_dqn_lr.png",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Number of obstacles
    # ─────────────────────────────────────────────────────────────────────────
    print("  Sweeping n_obstacles …")
    obs_counts = [6, 10, 14, 18, 22]
    obs_res: Dict[str, Dict] = {
        "Q-Learning": {n: [] for n in obs_counts},
        "DQN":        {n: [] for n in obs_counts},
        "PPO":        {n: [] for n in obs_counts},
    }
    for n in obs_counts:
        for s in sweep_seeds:
            env_cfg = {**base_cfg, "seed": base_cfg["seed"] + s}
            env = GridWorldEnv(
                grid_size=GRID_SIZE, n_obstacles=n,
                seed=env_cfg["seed"], dynamic_fraction=0.0,
            )
            obs_res["Q-Learning"][n].append(
                _sweep_metric(train_q_learning(env, n_episodes=N_SW, seed=s))
            )
            env = GridWorldEnv(
                grid_size=GRID_SIZE, n_obstacles=n,
                seed=env_cfg["seed"], dynamic_fraction=0.0,
            )
            obs_res["DQN"][n].append(
                _sweep_metric(train_dqn(env, n_episodes=N_SW, seed=s))
            )
            env = GridWorldEnv(
                grid_size=GRID_SIZE, n_obstacles=n,
                seed=env_cfg["seed"], dynamic_fraction=0.0,
            )
            obs_res["PPO"][n].append(
                _sweep_metric(train_ppo(env, n_episodes=N_SW, seed=s))
            )

    plot_parameter_sweep(
        obs_res,
        param_name="Number of Obstacles",
        metric_name="Success Rate",
        title="Obstacle Count Sweep",
        path=f"{OUT_PLOTS}/sweep_obstacles.png",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Reward shaping comparison (shaped vs unshaped)
    # ─────────────────────────────────────────────────────────────────────────
    print("  Reward shaping comparison …")
    shaped_rewards:   Dict[str, List[float]] = {"Q-Learning": [], "DQN": [], "PPO": []}
    unshaped_rewards: Dict[str, List[float]] = {"Q-Learning": [], "DQN": [], "PPO": []}

    N_SHAPE = 300
    for s in sweep_seeds:
        env_s  = _make_env({**base_cfg, "seed": base_cfg["seed"] + s}, shaped=True)
        env_u  = _make_env({**base_cfg, "seed": base_cfg["seed"] + s}, shaped=False)

        shaped_rewards["Q-Learning"]   += train_q_learning(env_s, n_episodes=N_SHAPE, seed=s).rewards
        unshaped_rewards["Q-Learning"] += train_q_learning(env_u, n_episodes=N_SHAPE, seed=s).rewards

        env_s = _make_env({**base_cfg, "seed": base_cfg["seed"] + s}, shaped=True)
        env_u = _make_env({**base_cfg, "seed": base_cfg["seed"] + s}, shaped=False)
        shaped_rewards["DQN"]   += train_dqn(env_s, n_episodes=N_SHAPE, seed=s).rewards
        unshaped_rewards["DQN"] += train_dqn(env_u, n_episodes=N_SHAPE, seed=s).rewards

        env_s = _make_env({**base_cfg, "seed": base_cfg["seed"] + s}, shaped=True)
        env_u = _make_env({**base_cfg, "seed": base_cfg["seed"] + s}, shaped=False)
        shaped_rewards["PPO"]   += train_ppo(env_s, n_episodes=N_SHAPE, seed=s).rewards
        unshaped_rewards["PPO"] += train_ppo(env_u, n_episodes=N_SHAPE, seed=s).rewards

    plot_reward_shaping(
        shaped_rewards, unshaped_rewards,
        path=f"{OUT_PLOTS}/reward_shaping_comparison.png",
    )
    print("  Sweeps complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# multi-experiment aggregate plots
# ═══════════════════════════════════════════════════════════════════════════════
def plot_aggregate(
    all_exp_results: List[Tuple[Dict[str, TrainingHistory], List[EvalResult]]],
) -> None:
    _pp("Aggregate Plots")

    # Stacked learning curves (all experiments, Q-Learning only for clarity)
    for method in ["Q-Learning", "DQN", "PPO"]:
        fig_data = {}
        for (hists, _), cfg in zip(all_exp_results, EXPERIMENTS):
            if method in hists:
                fig_data[cfg["label"]] = hists[method].rewards
        if fig_data:
            plot_learning_curves(
                fig_data,
                title=f"{method} — Reward Across Experiments",
                path=f"{OUT_PLOTS}/agg_{method.lower().replace('-','_')}_learning.png",
            )

    # Aggregate comparison bar chart
    plot_aggregate_comparison(
        [er for _, er in all_exp_results],
        exp_labels=[cfg["label"] for cfg in EXPERIMENTS],
        path=f"{OUT_PLOTS}/aggregate_comparison.png",
    )

    # Difficulty progression chart (how each method degrades as env gets harder)
    diff_data = [
        {
            "label": cfg["label"],
            "tag":   cfg["tag"],
            "eval_results": eval_res,
        }
        for (_, eval_res), cfg in zip(all_exp_results, EXPERIMENTS)
    ]
    plot_difficulty_progression(
        diff_data,
        path=f"{OUT_PLOTS}/difficulty_progression.png",
    )

    # CSV / JSON export
    all_rows: List[Dict] = []
    for (hists, eval_res), cfg in zip(all_exp_results, EXPERIMENTS):
        for r in eval_res:
            row = r.as_dict()
            row["experiment"] = cfg["label"]
            all_rows.append(row)

    # Collect the actual EvalResult objects (not re-constructed from dicts)
    all_eval_results: List[EvalResult] = [r for _, evals in all_exp_results for r in evals]
    write_eval_csv(all_eval_results, f"{OUT_TABS}/evaluation_summary.csv")

    summary = {}
    for (hists, eval_res), cfg in zip(all_exp_results, EXPERIMENTS):
        entry: Dict = {}
        for method, h in hists.items():
            entry[method] = {
                "final_success_rate":  round(h.success_rate(), 4),
                "convergence_episode": h.convergence_episode(),
                "avg_reward_last50":   round(h.avg_reward(), 2),
                "avg_steps_last50":    round(h.avg_steps(), 2),
            }
        for r in eval_res:
            if r.method not in entry:   # Dijkstra has no training history
                entry[r.method] = {}
            entry[r.method]["eval"] = r.as_dict()
        summary[cfg["label"]] = entry

    write_json(summary, f"{OUT_TABS}/results_summary.json")
    print(f"  Tables saved → {OUT_TABS}/")


# ═══════════════════════════════════════════════════════════════════════════════
# main entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    total_t0 = time.time()
    print("\n" + "█" * 60)
    print("  RL Navigation — Full Experiment Pipeline")
    print("█" * 60)

    all_results: List[Tuple[Dict[str, TrainingHistory], List[EvalResult]]] = []

    # 1. Run the three core experiments
    for cfg in EXPERIMENTS:
        hists, evals = run_experiment(cfg)
        all_results.append((hists, evals))

    # 2. Aggregate plots + tables
    plot_aggregate(all_results)

    # 3. Parameter sweeps
    run_parameter_sweeps()

    total_elapsed = time.time() - total_t0
    _pp(f"All done in {total_elapsed/60:.1f} min")
    print(f"  Plots      → {OUT_PLOTS}/")
    print(f"  Animations → {OUT_ANIM}/")
    print(f"  Tables     → {OUT_TABS}/")


if __name__ == "__main__":
    main()
