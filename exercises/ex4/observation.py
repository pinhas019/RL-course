"""
Assignment 4 — Observation function.

Implemented alternative: **B — egocentric observation (centered on the agent)**.

At every step the agent receives a deterministic 3x3 window of the board in
which the agent occupies the CENTER cell.  The window is sliced directly out
of the full board representation known to the system (walls / goals from the
map, boxes from the fully-known current box configuration) around the agent's
true — but hidden from the agent itself — position, exactly as required by
the assignment.  No MiniGrid wrapper (e.g. ViewSizeWrapper) is used: MiniGrid's
built-in observation is a forward-facing, direction-dependent window and
matches neither alternative.

Why alternative B?
------------------
* It is symmetric: the agent senses one cell in every direction, so a single
  observation constrains its position from all four sides simultaneously —
  walls, boxes and goals to the south/east/west are just as informative as
  those to the north.  Alternative A only ever senses the region north of the
  agent, so on maps whose distinguishing features lie south of the agent
  (e.g. the goal row at the bottom of our maps) localization is much slower.
* It is well-defined everywhere: for an agent standing next to the top wall,
  a "window located north of the agent" consists almost entirely of
  out-of-board cells and carries almost no information.
* It is independent of the agent's heading, which matches this environment's
  dynamics where the stochastic move can slide the agent sideways: after an
  unexpected lateral deviation the egocentric window immediately reveals the
  local surroundings in the deviation direction.

Determinism: O(o | s', a) = 1 iff o is exactly the 3x3 window around s',
otherwise 0.  There is no sensor noise — the only uncertainty in the problem
is the unknown initial position combined with the stochastic dynamics.

Observation content per cell (assignment: which cells are free, which are
walls, and whether there is a box):

    'W' wall (or out of board)   'B' small box   'H' heavy box
    'G' goal cell (uncovered)    '.' free cell

Other agents are NOT part of the observation: the sensor reads the board
(static structure + boxes), and box/world knowledge is global anyway.  This
keeps O(o | s') a deterministic function of the observing agent's own
position, as the assignment specifies.
"""

# Window offsets in row-major order: (dx, dy) for the 3x3 egocentric window.
WINDOW_OFFSETS = tuple(
    (dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
)


def observe(model, pos, smalls, heavies):
    """
    Deterministic egocentric observation for a single agent.

    Parameters
    ----------
    model   : WorldModel  (static walls / goals)
    pos     : (x, y) the agent's TRUE position (hidden from the agent)
    smalls  : frozenset of current small-box cells
    heavies : frozenset of current heavy-box cells

    Returns
    -------
    tuple of 9 one-character labels, row-major, agent at index 4.
    """
    cells = []
    for dx, dy in WINDOW_OFFSETS:
        c = (pos[0] + dx, pos[1] + dy)
        if (c[0] < 0 or c[0] >= model.width or c[1] < 0 or c[1] >= model.height
                or c in model.walls):
            cells.append('W')
        elif c in smalls:
            cells.append('B')
        elif c in heavies:
            cells.append('H')
        elif c in model.goal_set:
            cells.append('G')
        else:
            cells.append('.')
    return tuple(cells)


def joint_observation(model, positions, smalls, heavies):
    """Tuple of per-agent egocentric observations (one entry per robot)."""
    return tuple(observe(model, p, smalls, heavies) for p in positions)


def consistent_cells(model, obs, smalls, heavies):
    """
    Full-map-knowledge helper for particle-depletion handling: return every
    free cell whose deterministic window equals `obs`.  Because the
    observation function is deterministic, these are exactly the positions
    with O(obs | s) = 1.
    """
    return [
        c for c in model.free_cells
        if c not in smalls and c not in heavies
        and observe(model, c, smalls, heavies) == obs
    ]
