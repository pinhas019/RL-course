# Assignment 4 — Planning under Partial Observability: POMDP & POMCP

**Students:** Pinhas Aburmad 212146849 & Segev Olpak 325176188

The Box Pushing problem of the previous assignments, with a new layer of
uncertainty: the robot does **not** know its initial position on the board and
must estimate it while acting, based only on partial 3×3 observations. The
problem is a POMDP and is solved online with **POMCP** (Partially Observable
Monte-Carlo Planning, Silver & Veness 2010) — implemented entirely from
scratch (no ready-made RL / planning / POMDP libraries).

## Files

| File | Content |
|------|---------|
| [`observation.py`](observation.py) | The observation function (Alternative B — egocentric 3×3 window) |
| [`particle_filter.py`](particle_filter.py) | Particle filter for the belief state (rejection sampling + depletion handling) |
| [`pomcp.py`](pomcp.py) | POMCP: MCTS over particles, UCB1, heuristic rollouts, enforced time budget |
| [`world_model.py`](world_model.py) | Generative model G(s,a) → (s′,o,r) mirroring the Assignment-2 stochastic dynamics |
| [`run_experiments.py`](run_experiments.py) | Online-planning loop + full experiment protocol |
| [`results.txt`](results.txt) | Raw experiment logs |

## How to run

```bash
source .venv/bin/activate          # env with minigrid / pettingzoo / numpy

# Full assignment protocol: both scenarios × both budgets (1s, 20s) × 30 runs
python3 exercises/ex4/run_experiments.py

# Or selectively, e.g. only the 20-second-budget experiments:
python3 exercises/ex4/run_experiments.py --budgets 20 --n-runs 30
```

## The POMDP

* **States** — identical to the previous assignments, except that the agent's
  own position is unknown. Per robot, the space of possible positions is the
  set of free cells of the board. Box locations and all other world
  components remain fully known; directions (headings) are deterministic
  given the executed actions, so the *belief* is only over positions.
* **Actions / transitions** — identical to Assignment 2:
  move 0.8 / 0.1 / 0.1 (side deviations), push 0.8 / 0.2, unmet preconditions
  are deterministic no-ops. Primitive per-agent actions are
  rotate-left / rotate-right / forward; in the two-robot scenario the planner
  searches over the 9 **joint** actions.
* **Observations** — deterministic 3×3 window (see below):
  O(o | s′, a) = 1 iff o is exactly the window around s′, else 0. No sensor
  noise; the only uncertainty is the unknown initial position combined with
  the stochastic dynamics.
* **Reward / termination / γ** — as in the previous assignments: reward 1.0
  when *all* goal cells are covered by boxes (episode terminates), γ = 0.95.

## 1. Observation function — chosen alternative: **B (egocentric)**

Implemented in [`observation.py`](observation.py) as our own function that
slices the full board representation known to the system directly around the
robot's true (but hidden from the robot) position — no MiniGrid wrapper is
involved (MiniGrid's built-in view is forward-facing and direction-dependent,
matching neither alternative; none of its observation components are used).
Each of the 9 cells is labeled: `W` wall, `B` small box, `H` heavy box,
`G` goal, `.` free.

**Why B and not A (window fixed to the north):**

1. **Symmetric information.** The egocentric window senses one cell in every
   direction, so a single observation constrains the position from all four
   sides — walls, boxes and goals south of the robot (e.g. the goal row at
   the bottom of our maps) are just as informative as those to the north.
   Alternative A never sees anything south of the robot, which slows
   localization exactly where the task happens (near the goal row).
2. **Well-defined everywhere.** Next to the top wall, a "window north of the
   agent" lies almost entirely outside the board and carries almost no
   information; the egocentric window always contains real board content.
3. **Matches the dynamics.** The stochastic move can slide the robot
   sideways; after such a deviation the egocentric window immediately reveals
   the surroundings in the deviation direction, so the particle filter
   corrects the belief within a step or two.

Other robots are not part of the observation content (the assignment defines
the content as which cells are free / walls / boxes), which keeps
O(o | s′) a deterministic function of the observing robot's own position.

## 2. Particle filter

Implemented in [`particle_filter.py`](particle_filter.py).

* **Representation** — N = 500 particles; each particle is a position
  hypothesis. In the two-robot scenario a particle is a *joint* hypothesis
  ((x₀,y₀),(x₁,y₁)), which keeps the coupling introduced by the heavy-box
  push (both robots must occupy the same cell) exact.
* **Initialization** — uniform over all free cells of the board, since the
  robot does not know its starting position.
* **Update** — after executing real action *a* and receiving real observation
  *o*: unweighted **rejection sampling** exactly as in the original POMCP
  paper — sample a particle, simulate *a* on it through the same generative
  model used for planning (transition + observation), accept iff the
  simulated observation equals *o* (the fully-known new box configuration
  must match as well), repeat until N new particles are collected.
* **Particle depletion** — because the observation function is deterministic
  the rejection rate can be high, so the number of sampling attempts is
  capped (100·N). If the filter is still short of N particles, it is
  replenished by **reinvigoration that exploits full map knowledge**: the
  deterministic observation function is inverted directly — every free cell
  whose 3×3 window equals the received observation is consistent with it —
  and fresh particles are drawn uniformly from those cells. This also makes
  the very first update cheap: after the first observation the belief
  collapses at once to the (usually few) cells consistent with it.
* The same particle set serves both as the belief between environment steps
  and as the root belief from which POMCP samples its simulation start
  state at every single simulation (the search tree itself carries no
  per-node particle sets — only visit counts and Q values — since it is
  rebuilt from scratch every decision and only ever needs the root belief
  to sample from).

## 3. POMCP

Implemented in [`pomcp.py`](pomcp.py), following Silver & Veness (2010):

* Every simulation starts from a state **sampled from the particle filter**
  (never from an explicit distribution).
* Inside the known tree (visited action–observation histories) actions are
  selected with **UCB1**: Q(h,a) + c·√(ln N(h) / N(h,a)).
* On leaving the tree, exactly one new node is added and its value is
  estimated with a **rollout** down to the search horizon. The rollout
  policy is ε-greedy (ε = 0.2) around a simple domain heuristic: walk (via
  BFS distances on the static grid) to the staging cell behind the nearest
  unplaced box and push it toward its nearest free goal; when only the heavy
  box remains both robots converge to the *same* staging cell, which
  produces the cooperative push. (The reward itself is the sparse terminal
  reward, as defined; the heuristic is only the rollout policy, as permitted.)
* Returns are **backed up** along the simulated path, updating visit counts
  and running-average values.
* **The time budget is enforced in the search loop itself**: new simulations
  are launched only while `time.perf_counter() − t₀ < budget`, so every
  decision is returned within the allotted 1s / 20s (each simulation is
  depth-bounded, so overshoot is at most a single simulation, ~1 ms).
* After the budget is exhausted, the real action with the highest estimated
  root value is executed; after the real observation arrives the particle
  filter is updated and planning starts afresh at the next step.

Multi-robot planning is centralized: joint actions (3² = 9), joint
observations, joint particles — this is what lets the search discover the
simultaneous same-cell heavy-box push.

## Hyper-parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `time_budget_short` | 1 s | enforced inside the search loop |
| `time_budget_long` | 20 s | enforced inside the search loop |
| `n_runs` | 30 | per scenario × budget |
| `n_particles` | 500 | recommended starting point; no depletion observed thanks to the map-based reinvigoration, so it was not increased |
| `max_depth` (horizon) | 60 | tree + rollout; γ⁶⁰ ≈ 0.05, and ~60 steps suffice to solve either map from any hypothesis, so deeper search adds ≈ nothing |
| UCB1 `c` | 1.0 | recommended starting point; worked well |
| `gamma` | 0.95 | as in the previous assignments |
| `rollout_epsilon` | 0.2 | random-action probability inside the heuristic rollout |
| `max_steps` | 500 | episode cap, as in Assignment 2 |
| rejection-sampling cap | 100·N attempts | then map-based reinvigoration |

Deviations from the recommended values: none (all recommended values were
kept; the only additions are `rollout_epsilon` and the attempts cap, listed
above).

## Results

Mean ± std of the number of environment steps to solve the task, over 30
runs per cell (`results.txt` holds the raw per-run logs).

| Scenario | Budget 1 s | Budget 20 s |
|----------|-----------|-------------|
| Single agent | 16.53 ± 2.99 (30/30 solved) | 16.43 ± 2.87 (30/30 solved) |
| Two robots | 40.40 ± 5.69 (30/30 solved) | 33.33 ± 5.19 (30/30 solved) |

Wall-clock times per cell (30 runs each) were 502s / 9,880s / 1,245s / 20,073s
respectively — within ~1–3% of `n_runs × mean_steps × budget`, confirming the
per-decision budget is tightly enforced end to end.

## Discussion

**Effect of the computation budget (1 s vs 20 s).** The budget determines
how many POMCP simulations back each decision (≈ thousands at 1 s, ≈ tens of
thousands at 20 s). With more simulations, UCB1 statistics at the root are
better converged, so the chosen actions waste fewer steps on detours,
redundant rotations, and premature pushes made while the belief is still
multimodal — the effect is visible in the two-robot scenario, where the
joint action space (9 actions per node) spreads the same simulation budget
much thinner, and coordinated behaviors (the same-cell heavy push) need
deep, consistent action sequences to show value: going from 1 s to 20 s
lowers the mean (40.40 → 33.33 steps) and its standard deviation
(5.69 → 5.19), with every run solved at both budgets — at 20 s the
coordinated heavy-box push is found slightly faster and more consistently,
while at 1 s it is still found reliably, just with a bit more wandering
before the rendezvous. In the single-agent scenario the problem is easy
enough that 1 s (~2,000 simulations per decision) is already at the
plateau: the measured results at 1 s (16.53 ± 2.99) and at 20 s
(16.43 ± 2.87) are statistically identical — beyond the plateau, extra
computation cannot help, because the remaining step count is dominated by
the irreducible stochasticity of the transitions (failed pushes and
sideways moves), not by decision quality.

**Effect of the multi-robot scenario.** The two-robot scenario is harder on
every axis: the belief is over *joint* positions (|free cells|² hypotheses
instead of |free cells|), the branching factor grows from 3 to 9, and the
task itself contains the heavy box, which requires both robots to occupy the
same cell and push in the same direction simultaneously — a low-probability
event under exploration, and one that a 0.8-success push can still fail.
Accordingly, the mean number of steps is larger than in the single-agent
case at both budgets — even with the generous 20 s budget, two robots need
about twice the steps of the single agent (33.33 vs 16.43), with a large
share of the extra steps spent on the rendezvous + coordinated-push phase.
The second robot does not halve the solution time — the coordination
overhead dominates the parallelism gain on this map. On the other hand,
localization is not harder per robot: each robot's 3×3 observations localize
it within a few steps exactly as in the single-agent case.

**Localization behaves as expected.** Starting from a uniform belief over
all free cells, the first observation typically collapses the belief to a
handful of consistent cells (the deterministic observation function is
inverted directly on the map during reinvigoration), and after 2–4 moves the
belief is usually a single hypothesis tracked through the stochastic
dynamics. Occasional stochastic deviations re-inflate the belief briefly;
the rejection-sampling update absorbs them within a step.
