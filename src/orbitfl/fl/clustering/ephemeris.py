"""Clustering scheduled in advance from orbital geometry (GTRFL in the thesis).

EphemerisClustering holds a membership schedule for every round, computed from
the trace before training starts, so it needs no discovery rounds. Two variants
exist:
  ground region  cluster = the ground region under the satellite at the round's
                 time. This is the proposed method and the one used in all
                 thesis experiments (strategy param use_meo_proximity: false).
  MEO proximity  cluster = the nearest visible MEO at the round's time.
"""

from __future__ import annotations

import numpy as np

from orbitfl.fl.clustering.base import ClusteringStrategy, FLState
from orbitfl.trace.reader import Trace


class EphemerisClustering(ClusteringStrategy):
    """Clustering from a precomputed, geometry-based schedule.

    The schedule is an (R, N_leo) int array indexed by round and is fixed before the
    first training round, so the strategy has zero discovery overhead.

    The constructor builds the MEO-proximity schedule (cluster id = index of the
    nearest visible MEO). The ground-region schedule needs the world's region map,
    which this module does not import; build that schedule outside and pass it to
    from_region_schedule (see experiments.runner).
    """

    def __init__(
        self,
        trace: Trace,
        round_duration_s: int,
        round_budget: int,
        use_meo_proximity: bool = True,
    ) -> None:
        """Precompute the MEO-proximity membership schedule for all rounds.

        Parameters
        ----------
        trace             : Trace   Loaded trace.
        round_duration_s  : int     Seconds per FL round (cfg.fl.round_duration_s).
        round_budget      : int     Total number of rounds (cfg.fl.round_budget).
        use_meo_proximity : bool    Must be True here. For region clusters use
                                    from_region_schedule instead.
        """
        self._trace = trace
        self._use_meo_proximity = use_meo_proximity

        # Round start times in seconds from the trace epoch
        round_times = np.arange(round_budget, dtype=float) * round_duration_s

        if use_meo_proximity:
            # (R, N_leo) int, meo_id or -1, taken from the trace assignment table.
            # cluster_id == meo_id, so cluster ids are 0 .. N_meo-1 and -1.
            self._schedule: np.ndarray = trace.membership_schedule(round_times)
        else:
            # The region schedule needs OccupancyWorld, which cannot be imported here without
            # a circular import. Use from_region_schedule.
            raise NotImplementedError(
                "Region-based ephemeris clustering must be built with "
                "EphemerisClustering.from_region_schedule()."
            )

        self._overhead = 0

    @classmethod
    def from_region_schedule(
        cls,
        schedule: np.ndarray,
        round_budget: int,
    ) -> "EphemerisClustering":
        """Construct from a caller-supplied (R, N_leo) region-id schedule.

        The caller maps each satellite's position at each round to a region index with
        the world's region map and passes the result here.

        Parameters
        ----------
        schedule     : (R, N_leo) int  Region id per satellite per round.
        round_budget : int             Must match schedule.shape[0].
        """
        obj: EphemerisClustering = object.__new__(cls)
        obj._schedule = schedule
        obj._overhead = 0
        obj._use_meo_proximity = False
        return obj

    # --- ClusteringStrategy interface ---

    def membership(self, round_idx: int, state: FLState) -> np.ndarray:
        """Look up the precomputed cluster assignment for this round.

        Parameters
        ----------
        round_idx : int      Current round (0-based).
        state     : FLState  Unused; the schedule is already known.

        Returns
        -------
        np.ndarray  (N_leo,) int  Cluster id; -1 for LEOs with no cluster this round.
        """
        idx = min(round_idx, self._schedule.shape[0] - 1)
        return self._schedule[idx].copy()

    @property
    def overhead_rounds(self) -> int:
        """Zero: the whole schedule is computed from the trace."""
        return self._overhead
