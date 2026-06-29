# Ex2 — Study Notes

## Problem
Box-pushing with **stochastic** dynamics:
- Move into empty cell: 0.8 intended dir, 0.1 slip left, 0.1 slip right. Blocked slip → stay put.
- Push (precondition met): 0.8 success, 0.2 no-op.
- Precondition not met → no-op.

Two approaches to compare.

---

## Part 1 — Online Planning (determinized replanning)

**Idea:** pretend env is deterministic → run classical PDDL planner → execute only the first action → observe → replan.

### Loop
1. Dump current state to PDDL.
2. Call Fast Downward (`solve_pddl`).
3. Take **first PDDL action** from the plan.
4. Expand it into primitive env actions per agent (rotations + forward) via `get_required_actions`.
5. Pad action queues at the FRONT so all agents' final forwards happen in the same env step.
6. Step env, observe, replan.

### Why front-pad?
Heavy box needs **both** agents to push forward in the *same* env step. If agent A's queue is `[forward]` and B's is `[turn, turn, forward]`, popping naively makes A push alone at step 1 (precondition fails). Front-padding A to `[None, None, forward]` aligns the forwards on step 3 → joint push succeeds (0.8).

### Trade-offs
- ✅ Works on any map without precomputation.
- ❌ Planner call every step → slow.
- ❌ Determinized planner ignores slip probabilities → high variance.

---

## Part 2 — Modified Policy Iteration (MPI)

**Idea:** build the full MDP, solve offline once, execute via policy lookup.

### Two distinct phases

| | Phase 1 (offline) | Phase 2 (eval) |
|---|---|---|
| Runs | **once** | 100 times |
| Calls `env.step`? | no | yes |
| Updates `V` or `π`? | yes | **no** — pure execution |
| Cost | expensive (MDP build + MPI) | cheap dict lookups |

### Phase 1: build + solve

1. **Transition model** (`build_transition_model`) — BFS from initial state, analytically computes `P(s'|s,a)` from the stochastic rules. Does **not** sample from the env.
2. **MPI loop**:
   - Partial policy evaluation: `k=15` sweeps of `V(s) ← Σ P(s'|s,π(s)) · [R + γV(s')]`
   - Policy improvement: `π(s) ← argmax_a Σ P(s'|s,a) · [R + γV(s')]`
   - Stop when π stops changing.

**MPI vs other methods**: k=1 ≈ value iteration; k=∞ = full policy iteration. Middle ground = faster convergence in practice.

### Phase 2: evaluation (read-only)
Run policy 100 times, count steps. **Nothing is updated.** Rollouts only exist to report mean/std.

### State abstraction (key tractability choice)
State = `(agent0_pos, agent1_pos, box0_pos, box1_pos, heavy_pos)` — **direction dropped**.
- Reason: agent can rotate freely; orientation doesn't change long-term value.
- Saves 16× (4 dirs × 2 agents).
- Agents and small boxes are **sorted** → halves space (they're interchangeable).
- At execution, `mpi_policy_fn` emits rotations on-the-fly to face the target direction before moving.

### Dead-end pruning
Box at `x==1` or `y==1` → collapse state to `"FAIL"` (absorbing, 0 reward). No agent can stand behind it to push it away.

---

## "Evaluation" terminology trap

Same word, two roles:
1. **Policy evaluation** (RL theory): the inner k sweeps in Phase 1. Updates `V`, not `π`.
2. **Empirical evaluation**: Phase 2 rollouts. Updates **nothing** — just measures.

---

## Results

| Algorithm | Mean steps | Std | Std / Mean |
|---|---|---|---|
| Online Planning | 171.52 | 124.80 | 0.73 |
| MPI | **30.38** | **4.19** | **0.14** |

(100 runs each, same map, same stochastic env.)

### Reading the numbers

**Mean — MPI is ~5.6× faster.**
MPI's policy is computed against the **true stochastic MDP**, so it picks actions that are good *in expectation* (it accounts for the 0.2 push-failure and 0.1/0.1 slip outcomes). Online planning's policy is the first step of a plan computed under the **wrong model** (everything succeeds 100%), so it often picks actions that are optimal-if-deterministic but fragile under noise.

**Std — MPI is ~30× more consistent.**
- MPI std (4.19) is small but nonzero: the *policy* is fixed and optimal, but the *environment* is still stochastic, so two rollouts of the same policy take different numbers of steps depending on how slips land. This is the **irreducible** variance.
- Online std (124.80) is huge: when an unlucky slip happens, the planner just replans assuming determinism *again* and can get into long sequences of failed pushes / extra moves before recovering. Bad runs can hit the 500-step cap; lucky runs finish in ~50.

**Std/Mean ratio (coefficient of variation)** is the cleanest single comparison: 0.73 vs 0.14. Online is "high-variance noisy," MPI is "tight and predictable."

### Why the gap is this large

1. **MPI knows about slips, online doesn't.** The MPI policy might prefer a route that's slightly longer in expectation but avoids configurations where a slip would push a box against a wall (a dead-end → `"FAIL"` state with value 0). Online planning will happily walk along the wall because in its model nothing ever goes wrong.
2. **No replanning latency for MPI.** Each MPI step is a dict lookup. Each online step requires a fresh PDDL call to Fast Downward (the 171.52 figure measures *env steps*, but wall-clock time per run is even more lopsided).
3. **Determinized plans assume the worst case rarely.** A 0.2 push-failure rate sounds small, but a plan with 10 pushes has only a 0.8^10 ≈ 11% chance of executing cleanly. Online planning lives in the other 89% most of the time.

### Things that *do* inflate MPI's count
- MPI's state abstraction drops agent direction, so the execution wrapper has to spend env steps **rotating** before each move. The 30.38 figure includes those rotation steps. The "logical" plan length is closer to ~15–20.
- Even so, MPI dominates online planning by a wide margin.

### Takeaway for a test answer
- MPI is **better in expectation** because it solves the right problem (the stochastic MDP).
- Online planning is **worse and noisier** because it solves the wrong problem (a determinized approximation), and only corrects by replanning *after* things go wrong.
- The variance gap (0.14 vs 0.73) is arguably the more telling result than the mean gap.

---

## Where the cost lives

- **Online**: per-step at runtime (PDDL call every env step).
- **MPI**: once, up front (MDP enumeration + MPI iterations). After that, runtime is just dict lookups.

Trade-off: MPI's offline cost is **map-specific** — change the map, rebuild everything. Online planning is map-agnostic.

---

# Q&A — Likely Exam Questions

Covers ex2 internals **plus** the surrounding modules the prof can ask about: `environment/stochastic_env.py`, `environment/pddl_extractor.py`, `planner/pddl_solver.py`, `pddl/domain.pddl`, `visualize_plan.py`.

---

## A. Stochastic environment (`environment/stochastic_env.py`)

**Q1. How does `step()` actually implement the stochasticity?**
Three passes per env step:
1. **Pass 1 — gather intents.** Rotations (`action 0`, `action 1`) are applied immediately and deterministically. Forwards (`action 2`) are recorded as an intent `{target_pos, dir, vec}`, *not* executed yet.
2. **Pass 2 — heavy-box resolution.** Groups forward intents whose target cell is a heavy box. A heavy push only happens if **≥2 agents** share the same origin and the same direction (precondition). If satisfied, the engine flips a single coin: `random() < push_success_prob (0.8)` → both agents and the box move; else everyone stays. Both agents' intents are then *consumed* (popped).
3. **Pass 3 — individual forwards.** For each remaining intent: if forward cell is a small box → coin flip for push success; if forward cell is empty → call `_sample_move_dir()` to pick intended/left/right with 0.8/0.1/0.1, then move only if the resolved cell is also clear.

**Q2. Why are heavy pushes resolved before individual moves?**
Otherwise two agents who both intend to push a heavy box could be partially moved or accidentally interpreted as small-box pushers / movers. Resolving heavy pushes first guarantees the joint-action semantics.

**Q3. What happens if a slip targets a wall or another box?**
Agent stays put silently — see `if actual_fwd_cell is None or actual_fwd_cell.can_overlap()` in Pass 3. There's no error; the slip just "wastes" a step.

**Q4. What's the reward structure?**
Sparse and terminal: `0` every step, then `+1` for every agent in the step where `_all_boxes_on_goals()` becomes true. Termination is shared (every agent terminates simultaneously). Truncation occurs only at `steps >= max_steps`.

**Q5. Could the agent ever step onto another agent?**
No — agents are placed back onto the grid at their own positions at the end of `step()` (the loop at the bottom), and forward moves check `can_overlap()` against the **current** grid which contains the *other* agents. Two agents end up on the same cell only via a successful joint heavy push (intentional).

---

## B. PDDL pipeline (`pddl/domain.pddl`, `pddl_extractor.py`, `pddl_solver.py`)

**Q6. What does the PDDL domain model?**
Three actions — `move`, `push-small`, `push-heavy` — with STRIPS-style preconditions and effects. Predicates: `agent-at`, `box-at`, `heavybox-at`, `clear`, `adj`, `move-dir`. The domain is **deterministic** — exactly the determinization assumption the online planner relies on.

**Q7. Why does `pddl_extractor.py` generate the `move-dir` predicate?**
To force `push-small` and `push-heavy` to push *in a straight line*. The precondition `(move-dir ?from ?boxloc ?d) ... (move-dir ?boxloc ?toloc ?d)` requires the agent→box and box→destination directions to match. Without `move-dir`, the planner could "push" around corners.

**Q8. How is the PDDL goal constructed in `generate_problem`?**
Boxes are mapped to goal locations by **scan order** — the i-th box must land on the i-th goal. This is a STRIPS workaround for the fact that "any box at any goal" isn't easy to encode without disjunction / numeric constraints. It can occasionally produce a longer plan than necessary because the assignment is fixed.

**Q9. What does `solve_pddl` actually call?**
`unified_planning` library wrapping **Fast Downward** as the backend (`OneshotPlanner(name="fast-downward")`). Returns a `plan` object whose `.actions` is a list of grounded `ActionInstance` objects (e.g. `push-heavy(a0, a1, loc_3_2, loc_3_3, loc_3_4, hbx_0, down)`).

**Q10. What's the difference between `pddl/domain.pddl` (static file) and the one generated by `pddl_extractor.generate_domain`?**
Almost none — but the *generated* domain adds the `direction` type and `move-dir` predicate. The static file is the simpler reference; the runtime uses the generated version that supports directional pushes.

---

## C. The translation helpers (`visualize_plan.py`)

**Q11. What does `extract_target_pos(pddl_action)` return?**
A dict `{agent_name: (x, y)}` giving the target cell each involved agent needs to *end up adjacent to*. For `move` / `push-small` the target is the agent's destination; for `push-heavy` both agents share the same target — the **box's current cell** (because both agents end up there after a successful joint push).

**Q12. What does `get_required_actions(env, agent, target_pos)` do?**
Computes the direction vector from current pos to `target_pos`, finds the matching MiniGrid direction, then **only rotates right** until aligned, and appends `forward`. So a 180° turn always takes 2 right rotations. It does not optimize rotation cost — this is intentional for simplicity.

**Q13. Why does it raise `ValueError` if the target is not adjacent?**
Because PDDL action targets are always adjacent cells by construction (preconditions require `adj`). If we got a non-adjacent target, something went wrong upstream — better to fail loudly. `solution_ex2.py` catches this and submits an empty action so the env step happens anyway.

---

## D. Part 1 — Online planning

**Q14. Why execute only the *first* PDDL action?**
Because the underlying env is stochastic: any later action in the plan assumes the previous ones succeeded as intended. After one real env step we may be in a state the planner didn't expect, so we replan from scratch.

**Q15. Why pad action queues at the front?**
A joint push (heavy box) requires both agents to send `forward` in the **same env step**. Different agents may need different numbers of rotation steps to align. Front-padding the shorter queue with `None`s makes both queues' last element (the forward) line up at the same env step. Back-padding would leave the early forward alone again — preconditions fail.

**Q16. Why does online planning have such high variance (std=124.80)?**
Determinized planning is **brittle to stochasticity**. When a push fails, the planner just regenerates the same plan; if pushes keep failing the agent loops on the same action. When slips chain (slip-into-wall → stay → retry → slip again), short detours stack into long ones. The std is high because rollouts split into "lucky / no slips" and "unlucky / lots of slips" regimes.

**Q17. Is online planning **optimal** under the stochastic dynamics?**
No. The first action it picks is optimal for a *deterministic surrogate* of the MDP, not for the real MDP. A truly optimal stochastic policy could deliberately avoid configurations where a slip leads to a dead-end (e.g. routing wide around a wall). Online planning has no notion of risk.

**Q18. Could you make it better without going full MPI?**
Yes — common upgrades: **most-likely outcome determinization with hindsight**, **all-outcomes determinization** (treat each random outcome as a separate deterministic action), or **RTDP** (real-time dynamic programming) that mixes planning + value estimates.

---

## E. Part 2 — MPI (`solution_ex2.py` lines 212–490)

**Q19. Why drop direction from the state?**
A 4-direction-per-agent factor → 16× more states. Direction never gates the *optimal value* of a state: an agent can always rotate in place, so any state is reachable as the "same" state regardless of orientation, with a few extra cheap rotation steps. We restore direction at execution time in `mpi_policy_fn` by emitting rotations on-the-fly. The trade-off: the policy is slightly **suboptimal** in step count because it can't reason about rotation cost — but it stays tractable.

**Q20. Why sort agents and small boxes inside the state?**
Agents and small boxes are **interchangeable** (the goal is "all boxes on goals" — labels don't matter). Sorting canonicalizes the state so `(a0, a1, b0, b1)` and `(a1, a0, b1, b0)` map to the same key. This halves the state space and also matters for correctness — without it, the policy would be defined on two equivalent states that the BFS would have explored separately and might disagree on.

**Q21. Why prune dead-ends (`bx[0]==1 or bx[1]==1` → `"FAIL"`)?**
A box at `x=1` is one cell from the left wall, with the wall directly to its left. To push it right you'd need to stand to its left — but that cell is a wall. So it's unreachable from any direction that could move it away. Same for `y=1`. These states have value 0 forever; collapsing them to one absorbing `"FAIL"` state shrinks the model and lets MPI assign them value 0 without bothering to enumerate transitions.

**Q22. Why compute transitions analytically via BFS instead of sampling?**
We have the rules in closed form: 0.8 / 0.1 / 0.1 for moves, 0.8 / 0.2 for pushes. Sampling would require many rollouts per (state, action) to estimate probabilities and would still be noisy. The closed-form approach gives **exact** transition probabilities and lets MPI converge to the **true** optimal policy (subject to our state abstraction).

**Q23. What's the action space size?**
4 directions per agent × 2 agents = **16 joint actions**. For each (state, joint_action) we store a list of `(prob, next_state, reward)` outcomes. Joint heavy push collapses into a single 2-branch event (0.8 success / 0.2 fail).

**Q24. What does MPI converge to?**
A stationary deterministic policy **π** that is greedy w.r.t. its own value function — i.e., the **optimal policy** of the MDP. Convergence is detected as "policy didn't change in the improvement step."

**Q25. How does MPI relate to value iteration and policy iteration?**
| k (inner sweeps) | Method |
|---|---|
| 1 | Value iteration |
| ∞ (eval to convergence) | Policy iteration |
| 1 < k < ∞ | **Modified Policy Iteration** |
We chose **k=15**. Intuition: PI's bottleneck is the inner full evaluation; truncating it sacrifices little because we'll re-evaluate anyway, but speeds up each outer iteration dramatically.

**Q26. Why discount γ < 1 if the reward is sparse and finite-horizon?**
γ=0.95 creates a gradient of value across distant states. Without discounting, every state that can eventually reach the goal would have the same value (`1.0`), and the argmax over actions would be ambiguous — many actions would all look "equally good." Discounting makes "reach the goal sooner" strictly better than "reach the goal later," producing a well-defined gradient that argmax can exploit.

**Q27. What guarantees that BFS terminates?**
The state space is finite (cells × cells × cells × cells × cells minus invalid configurations), and we use a `visited` set. Once every reachable state has been enumerated, the BFS queue empties.

**Q28. What's the role of `mpi_policy_fn` at runtime?**
It bridges between abstract states (no direction) and the physical env (direction matters):
1. Read the env, build abstract state.
2. Look up `policy[state]` → `(dir0, dir1)`.
3. For each agent, if `current_dir == target_dir` → send `forward (2)`; otherwise send the shortest rotation (`turn_left (0)` or `turn_right (1)`, with 180° defaulting to right).

**Q29. Why does `mpi_policy_fn` sort agents the same way as `get_state`?**
Because the policy was learned under the sorted convention. If we asked `policy[state]` with `(a0, a1) = (sorted_lo, sorted_hi)` but applied the resulting `(dir0, dir1)` to the original `(a0, a1)` labels, the wrong physical agent would get the wrong direction. Sorting both at lookup time and at execution time keeps them aligned.

---

## F. Conceptual / theory

**Q30. Is MPI's policy actually optimal for the real env?**
Optimal for the **abstract MDP** (where direction is dropped). On the real env it's near-optimal but pays a small overhead for runtime rotations. Still beats online by ~5×.

**Q31. When would online planning beat MPI?**
- Large or open-ended maps where the state space explodes (MPI's BFS would run out of memory).
- Maps that change between episodes (MPI would have to rebuild every time).
- Tasks where computing the MDP is harder than just calling a deterministic planner.

**Q32. Could you combine the two?**
Yes — e.g., use the PDDL planner to **seed** an initial policy for MPI, or use online planning as a fallback when MPI's policy lookup fails (state unseen because BFS missed an outcome due to stochastic divergence — though the closed-form transition enumeration in this assignment makes that impossible by construction).

**Q33. Why use BFS rather than DFS or Dijkstra for state enumeration?**
We're not searching for a path — we want **all reachable states**. BFS just provides a clean queue-driven enumeration; DFS works equally well. Dijkstra would be wrong because state distance under stochastic dynamics isn't a meaningful weight here.

**Q34. What's the smallest detail the prof might catch?**
- The reward `R(s,a,s')` is **on transition into a terminal state**, computed by `is_trans_terminal(ns)` in `build_transition_model`. So entering the goal pays `+1`; lingering in the goal pays `0` (terminal self-loops).
- `policy_stable` only checks **action equality**; if there are ties in Q-values, the chosen action can flip between equally-good options and MPI may not detect convergence. In practice this rarely happens.
- The `try/except ValueError` in `run_online_planning` ([solution_ex2.py:124-130](exercises/ex2/solution_ex2.py#L124-L130)) is a safety net for when the planner returns an action whose target isn't adjacent (e.g., due to PDDL parsing edge cases).
