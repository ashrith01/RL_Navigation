# RL Navigation Project — Autonomous Robot Navigation with Reinforcement Learning

> **Course**: Applied Neural Networks (ELET6303)  
> **Task**: Train an RL agent to navigate a 2-D grid world from start (0,0) to goal (9,9) while avoiding static and dynamic obstacles.

---

## Algorithms Implemented

| Method | Type | Key Features |
|--------|------|--------------|
| **Q-Learning** | Tabular RL | ε-greedy, TD(0) update, discrete state index |
| **DQN** | Deep RL (value-based) | Dueling architecture, Double DQN, Prioritized Experience Replay (PER), soft target updates |
| **PPO** | Deep RL (policy gradient) | Actor-Critic, GAE, clipped surrogate loss, entropy bonus |
| **Dijkstra** | Classical path planning | Optimal shortest path (priority queue), used as the performance upper bound on static maps |

---

## Environment

- **Grid**: 10 × 10 cells  
- **Start**: (0, 0) · **Goal**: (9, 9)  
- **Obstacles**: 14 per experiment (configurable)
- **State** (12-dim vector for DQN/PPO):
  - Normalised agent position (2)
  - Normalised goal direction (2)
  - 8-directional sensor readings — distance to nearest obstacle/wall (8)
- **Tabular state** (Q-Learning): flat row-major cell index (0–99)
- **Actions**: Up / Down / Left / Right (4)
- **Rewards**: +100 goal | −1 per step | −10 collision · optional Manhattan shaping

### Obstacle Dynamics

| Experiment | Dynamic Fraction | Motion Mode | Difficulty |
|------------|-----------------|-------------|------------|
| Exp 1 (seed=42)  | 0 %   | —       | Static baseline |
| Exp 2 (seed=189) | 30 %  | Patrol (bounce) | Mild dynamic |
| Exp 3 (seed=464) | 60 %  | Homing (chase agent) | Heavy dynamic |

---

## Architecture Details

### Dueling Double DQN + PER
```
Input (12) → Linear(128) + LayerNorm + ReLU → Linear(64) + LayerNorm + ReLU
                    ↓                                   ↓
            Value head  V(s)                   Advantage head  A(s,a)
                    ↓                                   ↓
                        Q(s,a) = V(s) + (A(s,a) − mean_a A(s,a))
```
- **Double DQN**: online net selects action; target net evaluates → removes overestimation bias  
- **PER**: SumTree sampling proportional to |TD-error|^0.6; IS weights correct sampling bias  
- **Soft update**: θ_target ← 0.005·θ_online + 0.995·θ_target (Polyak averaging)

### PPO Actor-Critic
```
Input (12) → Linear(128) + Tanh → Linear(128) + Tanh
                    ↓                        ↓
            Policy head π(a|s)         Value head V(s)
```
- **GAE** (λ=0.95) for advantage estimation  
- **Clipped surrogate**: L = −min(r·A, clip(r, 0.8, 1.2)·A)  
- **Entropy bonus** (coef=0.01) to maintain exploration  
- **Rollout**: 8 episodes → 4 update epochs → mini-batches of 64

---

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (~30–60 min on CPU)
python main.py
```

All outputs are written to `outputs/`:

```
outputs/
├── plots/           # 30+ PNG figures (200 dpi, presentation-ready)
│   ├── exp1_learning_curves.png
│   ├── exp1_success_curves.png
│   ├── exp1_method_comparison.png
│   ├── exp1_path_comparison.png
│   ├── exp1_q_policy.png
│   ├── exp1_dqn_policy.png
│   ├── exp1_summary_dashboard.png
│   ├── aggregate_comparison.png
│   ├── sweep_ql_alpha.png
│   ├── sweep_gamma.png
│   ├── sweep_epsilon_decay.png
│   ├── sweep_dqn_lr.png
│   ├── sweep_obstacles.png
│   ├── reward_shaping_comparison.png
│   └── …
│
├── animations/      # GIF animations
│   ├── exp1_q_learning.gif
│   ├── exp1_dqn.gif
│   ├── exp1_ppo.gif
│   ├── exp1_dijkstra.gif
│   ├── exp1_comparison.gif
│   └── …  (×3 experiments)
│
└── tables/
    ├── evaluation_summary.csv
    └── results_summary.json
```

---

## Parameter Study

| Parameter | Values Tested | Methods |
|-----------|--------------|---------|
| Learning rate α | 0.01, 0.05, 0.1, 0.3, 0.5 | Q-Learning |
| Discount γ | 0.80, 0.90, 0.95, 0.99 | Q-Learning, DQN |
| ε-decay | 0.980, 0.990, 0.995, 0.999 | Q-Learning, DQN |
| DQN lr | 1e-4, 5e-4, 1e-3, 5e-3 | DQN |
| # obstacles | 6, 10, 14, 18, 22 | Q-Learning, DQN, PPO |
| Reward shaping | shaped vs unshaped | All RL methods |

---

## Evaluation Metrics

- **Success rate** (% episodes reaching goal within horizon)
- **Average reward** per episode
- **Average steps** per episode
- **Average path length** (successful episodes only)
- **Convergence episode** (first episode reaching 90% rolling success rate)

---

## Key Results (Exp 1 — Static)

| Method | Success Rate | Avg Steps | Avg Reward | Convergence Ep |
|--------|-------------|-----------|------------|---------------|
| Dijkstra | 100% | optimal | best | — |
| Q-Learning | ~100% | ~20 | ~79 | ~80 |
| DQN | ~100% | ~19 | ~80 | ~150 |
| PPO | ~100% | ~20 | ~79 | ~200 |

*Dijkstra provides the optimal path; RL methods learn near-optimal policies.  
On dynamic environments (Exp 2 & 3), deep RL (DQN, PPO) outperforms tabular Q-Learning.*

---

## File Structure

```
rl_nav_project/
├── environment.py    # GridWorldEnv (state, actions, dynamics)
├── q_learning.py     # Tabular Q-Learning agent
├── dqn_agent.py      # Dueling Double DQN + PER
├── ppo_agent.py      # PPO with Actor-Critic + GAE
├── dijkstra.py       # Dijkstra's shortest-path planner
├── train.py          # Training loops for all RL agents
├── evaluate.py       # Evaluation helpers + Dijkstra runner
├── visualize.py      # Presentation-quality matplotlib plots
├── animate.py        # GIF animation generation
├── main.py           # Full experiment orchestration
└── requirements.txt
```
