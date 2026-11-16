"""Static clustering by orbital plane.

PlaneClustering assigns each satellite to its orbital plane for the whole run.
This is the partition used by plane-based hierarchical FL schemes such as
P-Orion and serves as the static baseline. It needs no discovery rounds.
"""

from __future__ import annotations

import numpy as np

from orbitfl.fl.clustering.base import ClusteringStrategy, FLState
from orbitfl.trace.reader import Trace


class PlaneClustering(ClusteringStrategy):
    """Static orbital-plane partition.

    The cluster id of every LEO is its orbital plane number, read from the trace at
    construction and never changed. Ids run 0 .. N_planes-1; there are no -1
    entries because every LEO has a plane.
    """

    def __init__(self, trace: Trace) -> None:
        """Read the plane ids from the trace.

        Parameters
        ----------
        trace : Trace  Loaded trace; plane_id is read from the positions at t = 0.
        """
        # (N_leo,) int, plane ids from the positions table at t_idx = 0
        self._assignments: np.ndarray = trace.leo_plane_ids.copy()

    # --- ClusteringStrategy interface ---

    def membership(self, round_idx: int, state: FLState) -> np.ndarray:
        """Return the plane partition (the same every round).

        Parameters
        ----------
        round_idx : int      Unused.
        state     : FLState  Unused.

        Returns
        -------
        np.ndarray  (N_leo,) int  Plane id per LEO satellite (0-based, no -1).
        """
        return self._assignments.copy()

    @property
    def overhead_rounds(self) -> int:
        """Zero: the partition is read from the trace."""
        return 0
