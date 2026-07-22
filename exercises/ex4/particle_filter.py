"""
Assignment 4 — Particle filter over the belief state.

Each particle is one hypothesis of the (joint) agent position:

    particle = ((x0, y0), (x1, y1), ...)      one cell per robot

Only agent positions are hidden — directions are deterministic and box
locations are fully known — so nothing else needs to be carried in a
particle.  In the two-robot scenario a particle is a *joint* hypothesis,
which keeps the coupling introduced by heavy-box pushes (both robots must be
in the same cell) exact.

Belief update (after executing real action `a` and receiving real
observation `o`) follows the original POMCP paper (Silver & Veness, 2010):
unweighted rejection sampling.  A particle is drawn from the current filter,
`a` is simulated on it through the SAME generative model used by POMCP for
planning (transition + observation), and the particle is accepted iff the
simulated observation equals the true observation `o` (the fully-known new
box configuration is checked as well).  This repeats until N new particles
are collected.

Particle depletion: because the observation function is deterministic the
rejection rate can be high and the filter can empty out completely.  As the
assignment suggests, the number of sampling attempts is capped, and the
filter is then replenished by *reinvigoration* that exploits the full
knowledge of the board map: the deterministic observation function is
inverted directly — every free cell whose 3x3 window equals the received
observation is a consistent hypothesis — and fresh particles are drawn
uniformly from those consistent cells.
"""

import random

from observation import observe, joint_observation, consistent_cells


class ParticleFilter:
    def __init__(self, model, n_particles=500, max_attempts_factor=100,
                 rng=None):
        self.model = model
        self.n_particles = n_particles
        self.max_attempts_factor = max_attempts_factor
        self.rng = rng if rng is not None else random.Random()
        self.particles = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_uniform(self, smalls, heavies):
        """
        Spread particles uniformly over all free board cells (per agent),
        as required when the agent does not know its initial position.
        """
        free = [c for c in self.model.free_cells
                if c not in smalls and c not in heavies]
        self.particles = [
            tuple(self.rng.choice(free) for _ in range(self.model.n_agents))
            for _ in range(self.n_particles)
        ]
        return self.particles

    # ------------------------------------------------------------------
    # Belief update
    # ------------------------------------------------------------------

    def update(self, action, real_obs, dirs_before,
               smalls_before, heavies_before, smalls_after, heavies_after):
        """
        Rejection-sampling belief update after a REAL environment step.

        Parameters
        ----------
        action        : joint action actually executed (tuple, one per agent)
        real_obs      : joint observation actually received from the sensor
        dirs_before   : agents' (known) headings before the step
        smalls_before / heavies_before : (known) box cells before the step
        smalls_after  / heavies_after  : (known) box cells after the step
        """
        model = self.model
        new_particles = []
        attempts = 0
        max_attempts = self.max_attempts_factor * self.n_particles

        while len(new_particles) < self.n_particles and attempts < max_attempts:
            attempts += 1
            positions = self.rng.choice(self.particles)
            state = (positions, dirs_before, smalls_before, heavies_before)
            (next_pos, _, sim_smalls, sim_heavies), _, _ = model.step(state, action)
            # Accept iff the simulated observation matches the real one; the
            # box configuration is fully known, so it must match too.
            if (sim_smalls == smalls_after and sim_heavies == heavies_after
                    and joint_observation(model, next_pos,
                                          sim_smalls, sim_heavies) == real_obs):
                new_particles.append(next_pos)

        if len(new_particles) < self.n_particles:
            self._reinvigorate(new_particles, real_obs,
                               smalls_after, heavies_after)

        self.particles = new_particles
        return self.particles

    def _reinvigorate(self, new_particles, real_obs, smalls, heavies):
        """
        Depletion handling: invert the deterministic observation function on
        the fully-known map — per agent, every free cell whose window equals
        the received observation is consistent — and fill the filter with
        uniform draws over those cells.
        """
        per_agent_cells = []
        for agent_obs in real_obs:
            cells = consistent_cells(self.model, agent_obs, smalls, heavies)
            if not cells:  # cannot happen for a real observation; be safe
                cells = [c for c in self.model.free_cells
                         if c not in smalls and c not in heavies]
            per_agent_cells.append(cells)

        while len(new_particles) < self.n_particles:
            new_particles.append(
                tuple(self.rng.choice(cells) for cells in per_agent_cells)
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def n_distinct(self):
        return len(set(self.particles))
