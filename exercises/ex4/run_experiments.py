"""
Assignment 4 — Experiment runner: POMCP on the Box Pushing POMDP.

Online planning loop (replaces the classical PDDL replanning loop of the
previous assignments — plan over a BELIEF state instead of a known state):

    obs, info = env.reset()
    particles = init_uniform_particles(free_cells)
    done = False
    while not done:
        # 1. Plan: run POMCP over the current belief state
        action = run_pomcp(particles, time_budget=time_limit)
        # 2. Execute the real action in the environment, get a real observation
        obs, reward, terminated, truncated, info = env.step(action)
        # 3. Update the belief state given the action and the observation
        particles = update_particles(particles, action, obs)
        done = terminated or truncated

The real environment is StochasticMultiAgentBoxPushEnv (Assignment 2
dynamics).  Its MiniGrid observation is NOT used: the true observation is
produced by our own deterministic observation function (observation.py),
sliced around the agent's true position — which is read from the simulator
only to emulate the physical sensor and is never given to the planner.

Usage
-----
    # Full assignment protocol (both scenarios, budgets 1s & 20s, 30 runs):
    python3 exercises/ex4/run_experiments.py

    # Selective / reduced runs:
    python3 exercises/ex4/run_experiments.py --scenarios single --budgets 1 --n-runs 30
    python3 exercises/ex4/run_experiments.py --budgets 20 --n-runs 30

Results are appended as they complete to exercises/ex4/results.txt.
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from world_model import WorldModel, LEFT, RIGHT
from observation import joint_observation
from particle_filter import ParticleFilter
from pomcp import POMCP

# ---------------------------------------------------------------------------
# Scenarios — same board family as the previous assignments (Assignment 2 map)
# ---------------------------------------------------------------------------

# Two-robot scenario: the exact Assignment 2 map (2 small boxes + 1 heavy box
# that requires a coordinated same-cell push, 3 goals).
TWO_AGENT_MAP = [
    "WWWWWWWW",
    "W  AA  W",
    "W B C  W",
    "W      W",
    "W   B  W",
    "W G G GW",
    "WWWWWWWW",
]

# Single-agent scenario: same board without the second robot and without the
# heavy box (a lone agent can never push it), 2 small boxes, 2 goals.
SINGLE_AGENT_MAP = [
    "WWWWWWWW",
    "W  A   W",
    "W B    W",
    "W      W",
    "W   B  W",
    "W G G  W",
    "WWWWWWWW",
]

SCENARIOS = {
    "single": SINGLE_AGENT_MAP,
    "multi": TWO_AGENT_MAP,
}


def read_boxes(env):
    """Read the (fully known) current box configuration from the simulator."""
    smalls, heavies = set(), set()
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "box":
                if getattr(cell, "box_size", "") == "heavy":
                    heavies.add((x, y))
                else:
                    smalls.add((x, y))
    return frozenset(smalls), frozenset(heavies)


def true_positions(env):
    """True agent cells — read ONLY to emulate the physical sensor."""
    return tuple(env.agent_positions[a] for a in env.possible_agents)


def run_episode(ascii_map, time_budget, seed, args):
    """One full episode of online POMCP planning. Returns (steps, solved)."""
    np.random.seed(seed)               # environment stochasticity
    rng = random.Random(seed)          # particle filter + POMCP randomness

    env = StochasticMultiAgentBoxPushEnv(ascii_map=ascii_map,
                                         max_steps=args.max_steps)
    env.reset()

    model = WorldModel(ascii_map, rng=rng)
    pf = ParticleFilter(model, n_particles=args.n_particles, rng=rng)
    planner = POMCP(model, gamma=args.gamma, ucb_c=args.ucb_c,
                    max_depth=args.max_depth,
                    rollout_epsilon=args.rollout_epsilon, rng=rng)

    agents = env.possible_agents
    # Known quantities: box configuration and (deterministic) headings.
    smalls, heavies = read_boxes(env)
    dirs = tuple(env.agent_dirs[a] for a in agents)

    # Belief initialization: uniform over all free cells of the board.
    pf.init_uniform(smalls, heavies)

    steps = 0
    solved = False
    while True:
        # 1. Plan over the current belief (time budget enforced inside).
        action = planner.search(pf.particles, dirs, smalls, heavies,
                                time_budget)

        # 2. Execute the real action.
        _, rewards, terms, truncs, _ = env.step(
            {a: action[i] for i, a in enumerate(agents)})
        steps += 1

        # The new box configuration is fully known.
        new_smalls, new_heavies = read_boxes(env)

        # Real observation from our sensor model (around the TRUE positions).
        real_obs = joint_observation(model, true_positions(env),
                                     new_smalls, new_heavies)

        # 3. Belief update given the executed action and the observation
        #    (simulated from the PRE-step headings and box configuration).
        pf.update(action, real_obs, dirs,
                  smalls, heavies, new_smalls, new_heavies)

        # Advance the known quantities: headings change deterministically.
        dirs = tuple(
            (d - 1) % 4 if a == LEFT else (d + 1) % 4 if a == RIGHT else d
            for d, a in zip(dirs, action)
        )
        smalls, heavies = new_smalls, new_heavies

        if any(terms.values()):
            solved = True
            break
        if any(truncs.values()):
            break

    return steps, solved


def run_experiment(scenario, time_budget, args, out):
    ascii_map = SCENARIOS[scenario]
    steps_all, solved_all = [], []
    t_start = time.time()

    log(out, f"\n=== scenario={scenario}  time_budget={time_budget}s  "
             f"n_runs={args.n_runs} ===")
    for run in range(args.n_runs):
        seed = args.seed + run
        steps, solved = run_episode(ascii_map, time_budget, seed, args)
        steps_all.append(steps)
        solved_all.append(solved)
        log(out, f"  run {run + 1:2d}/{args.n_runs}: steps={steps:3d}  "
                 f"solved={solved}")

    mean, std = float(np.mean(steps_all)), float(np.std(steps_all))
    log(out, f"  -> mean steps = {mean:.2f}   std = {std:.2f}   "
             f"solved {sum(solved_all)}/{args.n_runs}   "
             f"(wall time {time.time() - t_start:.0f}s)")
    return mean, std, sum(solved_all)


def log(out, msg):
    print(msg, flush=True)
    out.write(msg + "\n")
    out.flush()


def main():
    parser = argparse.ArgumentParser(description="Assignment 4 — POMCP experiments")
    parser.add_argument("--scenarios", nargs="+", default=["single", "multi"],
                        choices=list(SCENARIOS))
    parser.add_argument("--budgets", nargs="+", type=float, default=[1.0, 20.0],
                        help="planning time budgets per decision, seconds")
    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--n-particles", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--ucb-c", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=60,
                        help="search horizon for tree + rollout")
    parser.add_argument("--rollout-epsilon", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "results.txt"))
    args = parser.parse_args()

    with open(args.results_file, "a") as out:
        log(out, "#" * 70)
        log(out, f"# POMCP experiments — {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(out, f"# params: n_particles={args.n_particles} gamma={args.gamma} "
                 f"ucb_c={args.ucb_c} max_depth={args.max_depth} "
                 f"rollout_eps={args.rollout_epsilon} max_steps={args.max_steps} "
                 f"seed={args.seed}")

        summary = []
        for scenario in args.scenarios:
            for budget in args.budgets:
                mean, std, solved = run_experiment(scenario, budget, args, out)
                summary.append((scenario, budget, mean, std, solved))

        log(out, "\n" + "=" * 70)
        log(out, "SUMMARY")
        log(out, f"{'scenario':<10} {'budget':>8} {'mean steps':>12} "
                 f"{'std':>8} {'solved':>10}")
        log(out, "-" * 52)
        for scenario, budget, mean, std, solved in summary:
            log(out, f"{scenario:<10} {budget:>7.0f}s {mean:>12.2f} "
                     f"{std:>8.2f} {solved:>7d}/{args.n_runs}")


if __name__ == "__main__":
    main()
