"""
Assignment 4 — POMCP (Partially Observable Monte-Carlo Planning).

Online planner following Silver & Veness (2010):

* The search operates DIRECTLY on the particle representation of the belief:
  every planning iteration samples a start state s from the current particle
  filter and runs one simulation from it — no explicit probability
  distribution over states is ever built.
* Inside the already-visited part of the search tree (action-observation
  histories) actions are chosen by UCB1, balancing exploitation (high
  estimated value) and exploration (few visits).
* When the simulation leaves the tree, exactly one new node is added and the
  value from that point is estimated with a rollout, run with a simple
  heuristic policy down to a fixed search horizon (max_depth).
* The return is backed up along the visited path, updating visit counts and
  value estimates of every node the simulation passed through.
* The per-decision computation budget (1s / 20s) is ENFORCED inside the
  search loop: simulations are launched only while wall-clock time remains,
  so the decision is always returned within the allotted time.
* After the budget is exhausted the real action with the highest estimated
  value at the root is returned.

Everything (tree, UCB1, rollouts, generative model) is implemented from
scratch — no ready-made RL / planning / POMDP library is used.

The multi-robot case is handled by a centralized planner over JOINT actions
(the cross product of the per-agent primitives), joint observations and
joint particles, which lets the search discover coordinated behaviors such
as the simultaneous same-cell heavy-box push.
"""

import math
import random
import time
from itertools import product

from world_model import AGENT_ACTIONS, DIR_VEC, LEFT, RIGHT, FORWARD
from observation import joint_observation


class _BeliefNode:
    """A history (belief) node of the search tree."""
    __slots__ = ("N", "expanded", "actions")

    def __init__(self, n_actions):
        self.N = 0
        self.expanded = False
        self.actions = [_ActionNode() for _ in range(n_actions)]


class _ActionNode:
    __slots__ = ("N", "Q", "children")

    def __init__(self):
        self.N = 0
        self.Q = 0.0
        self.children = {}          # observation -> _BeliefNode


class POMCP:
    def __init__(self, model, gamma=0.95, ucb_c=1.0, max_depth=60,
                 rollout_epsilon=0.2, rng=None):
        self.model = model
        self.gamma = gamma
        self.ucb_c = ucb_c
        self.max_depth = max_depth
        self.rollout_epsilon = rollout_epsilon
        self.rng = rng if rng is not None else random.Random()
        self.joint_actions = list(product(AGENT_ACTIONS, repeat=model.n_agents))
        self.last_n_simulations = 0

    # ------------------------------------------------------------------
    # Planning entry point
    # ------------------------------------------------------------------

    def search(self, particles, dirs, smalls, heavies, time_budget):
        """
        Run POMCP for `time_budget` seconds over the given belief and return
        the joint action with the highest estimated root value.

        The budget is enforced here: new simulations start only while
        wall-clock time remains, so the total planning time never exceeds the
        budget by more than the duration of a single (depth-bounded)
        simulation.
        """
        t0 = time.perf_counter()
        root = _BeliefNode(len(self.joint_actions))
        root.expanded = True
        n_sims = 0

        while time.perf_counter() - t0 < time_budget:
            # Sample an initial state from the current particle filter.
            positions = self.rng.choice(particles)
            state = (positions, dirs, smalls, heavies)
            self._simulate(state, root, 0)
            n_sims += 1
        self.last_n_simulations = n_sims

        visited = [(a.Q, i) for i, a in enumerate(root.actions) if a.N > 0]
        if not visited:  # degenerate budget — fall back to a random action
            return self.joint_actions[self.rng.randrange(len(self.joint_actions))]
        best_q = max(q for q, _ in visited)
        best = [i for q, i in visited if q == best_q]
        return self.joint_actions[self.rng.choice(best)]

    # ------------------------------------------------------------------
    # Simulation (tree policy + expansion + rollout + backup)
    # ------------------------------------------------------------------

    def _simulate(self, state, node, depth):
        if depth >= self.max_depth:
            return 0.0

        if not node.expanded:
            # New history: add it to the tree and estimate its value once.
            node.expanded = True
            return self._rollout(state, depth)

        a_idx = self._ucb_select(node)
        anode = node.actions[a_idx]
        next_state, reward, done = self.model.step(state, self.joint_actions[a_idx])

        if done:
            ret = reward
        else:
            positions, _, smalls, heavies = next_state
            obs = joint_observation(self.model, positions, smalls, heavies)
            child = anode.children.get(obs)
            if child is None:
                child = _BeliefNode(len(self.joint_actions))
                anode.children[obs] = child
            ret = reward + self.gamma * self._simulate(next_state, child, depth + 1)

        # Back up statistics along the path.
        node.N += 1
        anode.N += 1
        anode.Q += (ret - anode.Q) / anode.N
        return ret

    def _ucb_select(self, node):
        """UCB1: every action is tried once, then Q + c*sqrt(ln N / n)."""
        untried = [i for i, a in enumerate(node.actions) if a.N == 0]
        if untried:
            return self.rng.choice(untried)
        log_n = math.log(node.N)
        best_idx, best_val = 0, -float("inf")
        for i, a in enumerate(node.actions):
            val = a.Q + self.ucb_c * math.sqrt(log_n / a.N)
            if val > best_val:
                best_idx, best_val = i, val
        return best_idx

    # ------------------------------------------------------------------
    # Rollout with a simple heuristic policy
    # ------------------------------------------------------------------

    def _rollout(self, state, depth):
        """
        Estimate the value of a leaf by playing an epsilon-greedy heuristic
        policy (BFS-guided box pushing) until the horizon or termination.
        """
        ret = 0.0
        discount = 1.0
        d = depth
        while d < self.max_depth:
            action = self._rollout_policy(state)
            state, reward, done = self.model.step(state, action)
            ret += discount * reward
            if done:
                break
            discount *= self.gamma
            d += 1
        return ret

    def _rollout_policy(self, state):
        return tuple(
            self.rng.choice(AGENT_ACTIONS)
            if self.rng.random() < self.rollout_epsilon
            else self._greedy_agent_action(state, i)
            for i in range(self.model.n_agents)
        )

    def _greedy_agent_action(self, state, i):
        """
        Heuristic per-agent policy: pick the nearest box that is not yet on a
        goal, walk (BFS distances on the static grid) to the staging cell
        behind it, face the push direction and push it toward its nearest
        free goal.  When only the heavy box remains, both agents converge to
        the SAME staging cell, which produces the cooperative push.
        """
        model = self.model
        positions, dirs, smalls, heavies = state
        pos, heading = positions[i], dirs[i]

        free_goals = [g for g in model.goals if g not in smalls and g not in heavies]
        targets = [b for b in smalls if b not in model.goal_set]
        if not targets and model.n_agents >= 2:
            targets = [b for b in heavies if b not in model.goal_set]
        if not targets or not free_goals:
            return self.rng.choice(AGENT_ACTIONS)

        box = min(targets, key=lambda b: model.dist(pos, b))
        goal = min(free_goals, key=lambda g: model.dist(box, g))

        # Push direction: the box destination must be passable and must bring
        # the box closer to the goal; the agent stages on the opposite side.
        best_dir, best_dist = None, model.dist(box, goal)
        for d, (vx, vy) in enumerate(DIR_VEC):
            dest = (box[0] + vx, box[1] + vy)
            staging = (box[0] - vx, box[1] - vy)
            if (model.passable(dest, smalls, heavies)
                    and model.passable(staging, smalls, heavies)
                    and model.dist(dest, goal) < best_dist):
                best_dir, best_dist = d, model.dist(dest, goal)
        if best_dir is None:
            return self.rng.choice(AGENT_ACTIONS)

        vx, vy = DIR_VEC[best_dir]
        staging = (box[0] - vx, box[1] - vy)

        if pos == staging:
            desired = best_dir              # face the box and push
        else:
            # Step toward the staging cell along BFS distances.
            desired, step_dist = None, model.dist(pos, staging)
            for d, (nx, ny) in enumerate(DIR_VEC):
                nbr = (pos[0] + nx, pos[1] + ny)
                if (model.passable(nbr, smalls, heavies)
                        and model.dist(nbr, staging) < step_dist):
                    desired, step_dist = d, model.dist(nbr, staging)
            if desired is None:
                return self.rng.choice(AGENT_ACTIONS)

        if heading == desired:
            return FORWARD
        turn = (desired - heading) % 4
        return LEFT if turn == 3 else RIGHT
