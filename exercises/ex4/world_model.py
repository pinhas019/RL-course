"""
Assignment 4 — POMDP world model (generative model / black-box simulator).

This module implements a lightweight, fast copy of the dynamics of
``StochasticMultiAgentBoxPushEnv`` so that POMCP and the particle filter can
simulate thousands of transitions per second without touching MiniGrid.

State representation
--------------------
A (joint) simulation state is the tuple::

    State = (positions, dirs, smalls, heavies)

    positions : tuple[(x, y), ...]   one entry per agent (THE hidden part)
    dirs      : tuple[int, ...]      one entry per agent, 0=right 1=down 2=left 3=up
    smalls    : frozenset[(x, y)]    current small-box cells   (fully known)
    heavies   : frozenset[(x, y)]    current heavy-box cells   (fully known)

Only the agents' *positions* are hidden in this assignment.  Directions are
deterministic given the executed actions (rotations always succeed and moves
never change the heading), and box locations are declared fully known by the
assignment, so both are carried alongside the belief as known quantities.

Transition model (identical to Assignment 2 / StochasticMultiAgentBoxPushEnv)
-----------------------------------------------------------------------------
* move    : 0.8 intended direction, 0.1 deviate 90° left, 0.1 deviate 90°
            right; a deviation into a blocked cell leaves the agent in place.
* push    : 0.8 success, 0.2 no state change (small box: one pusher; heavy
            box: both agents pushing from the same cell in the same direction).
* If an action's precondition does not hold, it is a deterministic no-op.

Reward: 1.0 when ALL goal cells are covered by boxes (episode ends), else 0.
"""

import random
from collections import deque

# Per-agent primitive actions — same encoding as the environment.
LEFT, RIGHT, FORWARD = 0, 1, 2
AGENT_ACTIONS = (LEFT, RIGHT, FORWARD)

# 0=right, 1=down, 2=left, 3=up — matches minigrid.core.constants.DIR_TO_VEC
DIR_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))


class WorldModel:
    """Static world knowledge + generative model G(s, a) -> (s', r, done)."""

    def __init__(self, ascii_map, move_success_prob=0.8, push_success_prob=0.8,
                 rng=None):
        self.ascii_map = ascii_map
        self.width = len(ascii_map[0])
        self.height = len(ascii_map)
        self.move_success_prob = move_success_prob
        self.push_success_prob = push_success_prob
        self.rng = rng if rng is not None else random.Random()

        self.walls = set()
        self.goals = []
        self.init_smalls = set()
        self.init_heavies = set()
        self.init_positions = []          # true start cells (hidden from planner)
        for y, row in enumerate(ascii_map):
            for x, ch in enumerate(row):
                if ch == 'W':
                    self.walls.add((x, y))
                elif ch == 'G':
                    self.goals.append((x, y))
                elif ch == 'B':
                    self.init_smalls.add((x, y))
                elif ch == 'C':
                    self.init_heavies.add((x, y))
                elif ch == 'A':
                    self.init_positions.append((x, y))

        self.n_agents = len(self.init_positions)
        self.goal_set = frozenset(self.goals)
        # All non-wall cells; the belief over an agent's position is supported
        # on exactly this set ("free cells of the board" in the assignment).
        self.free_cells = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self.walls
        ]
        self._dist_cache = {}

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------

    def is_wall(self, pos):
        return pos in self.walls

    def passable(self, pos, smalls, heavies):
        """A cell an agent may stand on (agents never block each other)."""
        return pos not in self.walls and pos not in smalls and pos not in heavies

    def is_terminal(self, smalls, heavies):
        """True iff every goal cell is covered by a box (env termination)."""
        boxes = smalls | heavies
        return all(g in boxes for g in self.goal_set)

    # ------------------------------------------------------------------
    # Generative model
    # ------------------------------------------------------------------

    def step(self, state, joint_action):
        """
        Sample one stochastic transition, faithfully mirroring the pass
        structure of StochasticMultiAgentBoxPushEnv.step().

        Returns (next_state, reward, done).
        """
        positions, dirs, smalls, heavies = state
        positions = list(positions)
        dirs = list(dirs)
        smalls = set(smalls)
        heavies = set(heavies)
        rng = self.rng

        # ── Pass 1: rotations + forward intents ──────────────────────
        intents = {}                                    # agent idx -> (fwd, dir)
        for i, action in enumerate(joint_action):
            if action == LEFT:
                dirs[i] = (dirs[i] - 1) % 4
            elif action == RIGHT:
                dirs[i] = (dirs[i] + 1) % 4
            elif action == FORWARD:
                vec = DIR_VEC[dirs[i]]
                fwd = (positions[i][0] + vec[0], positions[i][1] + vec[1])
                intents[i] = (fwd, dirs[i])

        # ── Pass 2: heavy-box push resolution ────────────────────────
        pushes_by_box = {}
        for i, (fwd, d) in intents.items():
            if fwd in heavies:
                pushes_by_box.setdefault(fwd, []).append(i)
        for box_pos, pushers in pushes_by_box.items():
            if len(pushers) >= 2:
                origins = {positions[i] for i in pushers}
                push_dirs = {intents[i][1] for i in pushers}
                if len(origins) == 1 and len(push_dirs) == 1:
                    d = next(iter(push_dirs))
                    vec = DIR_VEC[d]
                    dest = (box_pos[0] + vec[0], box_pos[1] + vec[1])
                    if self.passable(dest, smalls, heavies):
                        if rng.random() < self.push_success_prob:
                            heavies.discard(box_pos)
                            heavies.add(dest)
                            for i in pushers:
                                positions[i] = box_pos
                # Intents of grouped heavy pushers are consumed either way.
                for i in pushers:
                    intents.pop(i, None)

        # ── Pass 3: individual forwards (small push / move), in agent order ──
        for i in sorted(intents):
            fwd, intended_dir = intents[i]
            if fwd in smalls:
                vec = DIR_VEC[intended_dir]
                behind = (fwd[0] + vec[0], fwd[1] + vec[1])
                if self.passable(behind, smalls, heavies):
                    if rng.random() < self.push_success_prob:
                        smalls.discard(fwd)
                        smalls.add(behind)
                        positions[i] = fwd
            elif self.passable(fwd, smalls, heavies):
                # Precondition met — apply directional stochasticity.
                r = rng.random()
                side = (1.0 - self.move_success_prob) / 2.0
                if r < self.move_success_prob:
                    actual_dir = intended_dir
                elif r < self.move_success_prob + side:
                    actual_dir = (intended_dir - 1) % 4
                else:
                    actual_dir = (intended_dir + 1) % 4
                vec = DIR_VEC[actual_dir]
                dest = (positions[i][0] + vec[0], positions[i][1] + vec[1])
                if self.passable(dest, smalls, heavies):
                    positions[i] = dest
                # else: deviated into an obstacle — the agent stays put.
            # else: wall / lone-agent heavy push — deterministic no-op.

        smalls = frozenset(smalls)
        heavies = frozenset(heavies)
        done = self.is_terminal(smalls, heavies)
        reward = 1.0 if done else 0.0
        return (tuple(positions), tuple(dirs), smalls, heavies), reward, done

    # ------------------------------------------------------------------
    # Shortest-path distances on the static (walls-only) grid
    # ------------------------------------------------------------------

    def dist_field(self, target):
        """BFS distance from every cell to `target`, ignoring boxes (memoized)."""
        field = self._dist_cache.get(target)
        if field is None:
            field = {target: 0}
            q = deque([target])
            while q:
                cx, cy = q.popleft()
                for vx, vy in DIR_VEC:
                    nxt = (cx + vx, cy + vy)
                    if nxt not in self.walls and nxt not in field:
                        field[nxt] = field[(cx, cy)] + 1
                        q.append(nxt)
            self._dist_cache[target] = field
        return field

    def dist(self, src, target):
        return self.dist_field(target).get(src, 10 ** 6)
