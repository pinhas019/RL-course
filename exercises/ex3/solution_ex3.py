"""
Exercise 3: Comparison of Tabular RL Algorithms on FrozenLake-v1 (8x8, Slippery)
=================================================================================

Submitted by:
Pinhas Aburmad 212146849
Segev Olpak 325176188

Implements and compares three tabular RL algorithms:
  1. Tabular Q-learning (off-policy TD control)
  2. Tabular SARSA       (on-policy TD control)
  3. REINFORCE            (tabular policy-gradient with softmax parameterisation)

All algorithms are trained for 500,000 environment steps on the stochastic
FrozenLake-v1 (8x8) and periodically evaluated by estimating V(s0).

Usage:
    python solution_ex3.py
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# ─────────────────────────────────────────────────────────────────────────────
# Global hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
GAMMA = 0.99
MAX_STEPS_PER_EPISODE = 200
TOTAL_TRAINING_STEPS = 500_000
SEEDS = [0, 1, 2, 3, 4]

# Q-learning
Q_ALPHA = 0.02
Q_EPSILON_START = 1.0
Q_EPSILON_END = 0.05
Q_EPSILON_DECAY_STEPS = 350_000

# SARSA
SARSA_ALPHA = 0.08
SARSA_EPSILON_START = 1.0
SARSA_EPSILON_END = 0.01
SARSA_EPSILON_DECAY_STEPS = 350_000

# REINFORCE
POLICY_LR = 0.005
THETA_INIT = 0.0
THETA_CLIP = 20.0

# Evaluation
EVAL_INTERVAL = 10_000
EVAL_EPISODES = 500

# Environment
N_STATES = 64   # 8x8
N_ACTIONS = 4   # left, down, right, up


# ─────────────────────────────────────────────────────────────────────────────
# Helper: epsilon schedule (linear decay)
# ─────────────────────────────────────────────────────────────────────────────
def get_q_epsilon(step: int) -> float:
    """Linear decay for Q-learning."""
    return max(
        Q_EPSILON_END,
        Q_EPSILON_START - step / Q_EPSILON_DECAY_STEPS * (Q_EPSILON_START - Q_EPSILON_END),
    )


def get_sarsa_epsilon(step: int) -> float:
    """Linear decay for SARSA."""
    return max(
        SARSA_EPSILON_END,
        SARSA_EPSILON_START - step / SARSA_EPSILON_DECAY_STEPS * (SARSA_EPSILON_START - SARSA_EPSILON_END),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: softmax with numerical stability
# ─────────────────────────────────────────────────────────────────────────────
def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities, subtracting max for numerical stability."""
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: make environment
# ─────────────────────────────────────────────────────────────────────────────
def make_env(seed: int) -> gym.Env:
    """Create a FrozenLake-v1 8x8 stochastic environment."""
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    env.reset(seed=seed)
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation routine (shared by all algorithms)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_policy(
    env: gym.Env,
    get_greedy_action,
    num_episodes: int = EVAL_EPISODES,
) -> float:
    """
    Estimate V(s0) by running `num_episodes` evaluation episodes with a
    frozen greedy policy.  Returns the mean discounted return G.

    Parameters
    ----------
    env : gym.Env
        The evaluation environment (same map, same dynamics).
    get_greedy_action : callable(state) -> int
        Returns the greedy action for a given state.
    num_episodes : int
        Number of evaluation episodes.

    Returns
    -------
    float
        Mean discounted return over all evaluation episodes.
    """
    returns = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        G = 0.0
        discount = 1.0
        for _ in range(MAX_STEPS_PER_EPISODE):
            action = get_greedy_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            G += discount * reward
            discount *= GAMMA
            state = next_state
            if terminated or truncated:
                break
        returns.append(G)
    return float(np.mean(returns))


# ═══════════════════════════════════════════════════════════════════════════════
# Algorithm 1: Tabular Q-learning
# ═══════════════════════════════════════════════════════════════════════════════
def train_q_learning(seed: int) -> Tuple[List[int], List[float]]:
    """
    Train Q-learning on FrozenLake-v1 (8x8, slippery).

    Returns
    -------
    eval_steps : list[int]
        Global step counts at which evaluations occurred.
    eval_values : list[float]
        Estimated V(s0) at each evaluation point.
    """
    rng = np.random.RandomState(seed)
    env = make_env(seed)
    eval_env = make_env(seed + 100)

    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)

    eval_steps: List[int] = []
    eval_values: List[float] = []

    global_step = 0
    next_eval_at = 0  # evaluate immediately at step 0

    while global_step < TOTAL_TRAINING_STEPS:
        state, _ = env.reset()

        for _ in range(MAX_STEPS_PER_EPISODE):
            # ── periodic evaluation ──
            if global_step >= next_eval_at:
                greedy = lambda s, Q=Q: int(np.argmax(Q[s]))
                v_s0 = evaluate_policy(eval_env, greedy)
                eval_steps.append(global_step)
                eval_values.append(v_s0)
                next_eval_at += EVAL_INTERVAL

            if global_step >= TOTAL_TRAINING_STEPS:
                break

            # ── epsilon-greedy action selection ──
            eps = get_q_epsilon(global_step)
            if rng.rand() < eps:
                action = rng.randint(N_ACTIONS)
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            global_step += 1

            # ── Q-learning update (off-policy) ──
            td_target = reward + GAMMA * np.max(Q[next_state]) * (1 - terminated)
            Q[state, action] += Q_ALPHA * (td_target - Q[state, action])

            state = next_state
            if terminated or truncated:
                break

    # Final evaluation at the end of training
    if len(eval_steps) == 0 or eval_steps[-1] != global_step:
        greedy = lambda s, Q=Q: int(np.argmax(Q[s]))
        v_s0 = evaluate_policy(eval_env, greedy)
        eval_steps.append(global_step)
        eval_values.append(v_s0)

    env.close()
    eval_env.close()
    return eval_steps, eval_values


# ═══════════════════════════════════════════════════════════════════════════════
# Algorithm 2: Tabular SARSA
# ═══════════════════════════════════════════════════════════════════════════════
def train_sarsa(seed: int) -> Tuple[List[int], List[float]]:
    """
    Train SARSA on FrozenLake-v1 (8x8, slippery).

    Returns
    -------
    eval_steps : list[int]
        Global step counts at which evaluations occurred.
    eval_values : list[float]
        Estimated V(s0) at each evaluation point.
    """
    rng = np.random.RandomState(seed)
    env = make_env(seed)
    eval_env = make_env(seed + 100)

    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)

    eval_steps: List[int] = []
    eval_values: List[float] = []

    global_step = 0
    next_eval_at = 0

    def eps_greedy_action(state: int, step: int) -> int:
        eps = get_sarsa_epsilon(step)
        if rng.rand() < eps:
            return rng.randint(N_ACTIONS)
        return int(np.argmax(Q[state]))

    while global_step < TOTAL_TRAINING_STEPS:
        state, _ = env.reset()
        action = eps_greedy_action(state, global_step)

        for _ in range(MAX_STEPS_PER_EPISODE):
            # ── periodic evaluation ──
            if global_step >= next_eval_at:
                greedy = lambda s, Q=Q: int(np.argmax(Q[s]))
                v_s0 = evaluate_policy(eval_env, greedy)
                eval_steps.append(global_step)
                eval_values.append(v_s0)
                next_eval_at += EVAL_INTERVAL

            if global_step >= TOTAL_TRAINING_STEPS:
                break

            next_state, reward, terminated, truncated, _ = env.step(action)
            global_step += 1

            # ── SARSA: choose next action with current policy ──
            next_action = eps_greedy_action(next_state, global_step)

            # ── SARSA update (on-policy) ──
            td_target = reward + GAMMA * Q[next_state, next_action] * (1 - terminated)
            Q[state, action] += SARSA_ALPHA * (td_target - Q[state, action])

            state = next_state
            action = next_action
            if terminated or truncated:
                break

    # Final evaluation
    if len(eval_steps) == 0 or eval_steps[-1] != global_step:
        greedy = lambda s, Q=Q: int(np.argmax(Q[s]))
        v_s0 = evaluate_policy(eval_env, greedy)
        eval_steps.append(global_step)
        eval_values.append(v_s0)

    env.close()
    eval_env.close()
    return eval_steps, eval_values


# ═══════════════════════════════════════════════════════════════════════════════
# Algorithm 3: REINFORCE (tabular softmax policy)
# ═══════════════════════════════════════════════════════════════════════════════
def train_reinforce(seed: int) -> Tuple[List[int], List[float]]:
    """
    Train REINFORCE (Monte-Carlo Policy Gradient) with a tabular softmax
    policy on FrozenLake-v1 (8x8, slippery).  No baseline is used.

    Returns
    -------
    eval_steps : list[int]
        Global step counts at which evaluations occurred.
    eval_values : list[float]
        Estimated V(s0) at each evaluation point.
    """
    rng = np.random.RandomState(seed)
    env = make_env(seed)
    eval_env = make_env(seed + 100)

    # Policy parameters θ(s, a) — a preference table
    theta = np.full((N_STATES, N_ACTIONS), THETA_INIT, dtype=np.float64)

    eval_steps: List[int] = []
    eval_values: List[float] = []

    global_step = 0
    next_eval_at = 0

    def sample_action(state: int) -> int:
        probs = softmax(theta[state])
        return int(rng.choice(N_ACTIONS, p=probs))

    def greedy_action(state: int) -> int:
        probs = softmax(theta[state])
        return int(np.argmax(probs))

    while global_step < TOTAL_TRAINING_STEPS:
        # ── periodic evaluation (before collecting an episode) ──
        if global_step >= next_eval_at:
            v_s0 = evaluate_policy(eval_env, greedy_action)
            eval_steps.append(global_step)
            eval_values.append(v_s0)
            next_eval_at += EVAL_INTERVAL

        # ── collect one episode ──
        episode_states = []
        episode_actions = []
        episode_rewards = []

        state, _ = env.reset()
        for _ in range(MAX_STEPS_PER_EPISODE):
            if global_step >= TOTAL_TRAINING_STEPS:
                break

            action = sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            global_step += 1

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state
            if terminated or truncated:
                break

        # ── compute discounted returns G_t for each time-step ──
        T = len(episode_rewards)
        if T == 0:
            continue

        G = np.zeros(T, dtype=np.float64)
        G[-1] = episode_rewards[-1]
        for t in range(T - 2, -1, -1):
            G[t] = episode_rewards[t] + GAMMA * G[t + 1]

        # ── policy gradient update ──
        for t in range(T):
            s_t = episode_states[t]
            a_t = episode_actions[t]
            probs = softmax(theta[s_t])

            # ∇_θ log π(a|s) = e_a − π(·|s)   (for softmax parameterisation)
            grad_log_pi = -probs.copy()
            grad_log_pi[a_t] += 1.0

            # θ(s, ·) ← θ(s, ·) + lr * γ^t * G_t * ∇ log π(a_t|s_t)
            theta[s_t] += POLICY_LR * (GAMMA ** t) * G[t] * grad_log_pi

        # ── clip θ for numerical stability ──
        np.clip(theta, -THETA_CLIP, THETA_CLIP, out=theta)

        # ── handle mid-episode evaluations for steps that passed thresholds ──
        while next_eval_at <= global_step and global_step < TOTAL_TRAINING_STEPS:
            v_s0 = evaluate_policy(eval_env, greedy_action)
            eval_steps.append(next_eval_at)
            eval_values.append(v_s0)
            next_eval_at += EVAL_INTERVAL

    # Final evaluation
    if len(eval_steps) == 0 or eval_steps[-1] < global_step:
        v_s0 = evaluate_policy(eval_env, greedy_action)
        eval_steps.append(global_step)
        eval_values.append(v_s0)

    env.close()
    eval_env.close()
    return eval_steps, eval_values


# ═══════════════════════════════════════════════════════════════════════════════
# Training driver: run all seeds for one algorithm
# ═══════════════════════════════════════════════════════════════════════════════
def run_algorithm(
    train_fn,
    algo_name: str,
) -> Dict[str, np.ndarray]:
    """
    Run `train_fn` across all SEEDS and aggregate evaluation curves.

    Returns
    -------
    dict with keys:
        'steps'      : 1-D array of evaluation step indices
        'mean'       : 1-D array of mean V(s0) across seeds
        'std'        : 1-D array of std  V(s0) across seeds
        'all_values' : 2-D array (n_seeds × n_evals) of raw values
    """
    all_steps = []
    all_values = []

    for i, seed in enumerate(SEEDS):
        print(f"  [{algo_name}] seed {seed}  ({i+1}/{len(SEEDS)})")
        steps, values = train_fn(seed)
        all_steps.append(steps)
        all_values.append(values)

    # Align all seeds to a common step grid (use the first seed's steps)
    # All runs use the same interval so step grids are nearly identical;
    # we truncate to the shortest common length for safety.
    min_len = min(len(v) for v in all_values)
    common_steps = np.array(all_steps[0][:min_len])
    aligned_values = np.array([v[:min_len] for v in all_values])  # (n_seeds, n_evals)

    mean_vals = np.mean(aligned_values, axis=0)
    std_vals = np.std(aligned_values, axis=0)

    return {
        "steps": common_steps,
        "mean": mean_vals,
        "std": std_vals,
        "all_values": aligned_values,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════
def plot_results(results: Dict[str, Dict[str, np.ndarray]], save_path: str) -> None:
    """
    Plot mean ± std of V(s0) vs. training steps for all algorithms.
    """
    plt.figure(figsize=(12, 7))

    colors = {
        "Q-learning": "#2196F3",
        "SARSA": "#FF9800",
        "REINFORCE": "#4CAF50",
    }

    for algo_name, data in results.items():
        steps = data["steps"]
        mean = data["mean"]
        std = data["std"]
        color = colors.get(algo_name, "gray")

        plt.plot(steps, mean, label=algo_name, color=color, linewidth=2)
        plt.fill_between(
            steps,
            mean - std,
            mean + std,
            alpha=0.2,
            color=color,
        )

    plt.xlabel("Environment Steps", fontsize=13)
    plt.ylabel("Estimated $V(s_0)$", fontsize=13)
    plt.title(
        "Comparison of Q-learning, SARSA, and REINFORCE\n"
        "on FrozenLake-v1 (8×8, Slippery)",
        fontsize=14,
    )
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  Exercise 3 — RL Algorithm Comparison on FrozenLake-v1 (8x8)")
    print("=" * 70)

    results = {}

    print("\n[1/3] Training Q-learning ...")
    results["Q-learning"] = run_algorithm(train_q_learning, "Q-learning")

    print("\n[2/3] Training SARSA ...")
    results["SARSA"] = run_algorithm(train_sarsa, "SARSA")

    print("\n[3/3] Training REINFORCE ...")
    results["REINFORCE"] = run_algorithm(train_reinforce, "REINFORCE")

    # ── Print final V(s0) summary ──
    print("\n" + "=" * 70)
    print("  Final V(s0) Summary")
    print("=" * 70)
    for algo_name, data in results.items():
        final_mean = data["mean"][-1]
        final_std = data["std"][-1]
        print(f"  {algo_name:12s}:  V(s0) = {final_mean:.4f} ± {final_std:.4f}")

    # ── Plot ──
    plot_path = "exercises/ex3/comparison_plot.png"
    plot_results(results, plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
