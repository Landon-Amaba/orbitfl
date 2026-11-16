"""Abstract clustering strategy interface.

A ClusteringStrategy provides
    membership(round_idx, state) -> (N_leo,) cluster ids for the round
    overhead_rounds              -> rounds spent discovering clusters so far
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from orbitfl.fl.aggregation import ClientUpdate, StateDict


@dataclasses.dataclass
class FLState:
    """Snapshot of the FL runner state passed to clustering strategies each round.

    Attributes
    ----------
    round_idx      : int              Current round index (0-based).
    global_model   : StateDict        Global model weights before this round.
    client_updates : list[ClientUpdate]
        All updates produced in the previous round (may be empty for round 0
        or when no clients participated). Similarity-based strategies read these
        to compute cluster memberships.
    round_start_models : dict[int, StateDict]
        Model each client in client_updates started local training from, keyed by
        sat_id. Empty when no updates exist. Read by SimilarityReactive when
        update_reference == "round_start".
    """

    round_idx: int
    global_model: "StateDict"
    client_updates: "list[ClientUpdate]"
    round_start_models: "dict[int, StateDict]" = dataclasses.field(default_factory=dict)


class ClusteringStrategy(ABC):
    """Abstract base for the clustering strategies.

    Subclasses implement membership() and overhead_rounds. In the membership array,
    -1 marks a LEO with no visible aggregator; the loop skips such clients.
    """

    @abstractmethod
    def membership(self, round_idx: int, state: FLState) -> np.ndarray:
        """
        Return cluster assignments for all LEO satellites this round.

        Parameters
        ----------
        round_idx : int      Current round (0-based).
        state     : FLState  Snapshot of the FL runner state.

        Returns
        -------
        np.ndarray  (N_leo,) int  Cluster id per LEO; -1 for unassigned.
        """

    @property
    @abstractmethod
    def overhead_rounds(self) -> int:
        """Number of rounds this strategy has spent on cluster discovery.

        Strategies that know their clusters in advance (ephemeris, plane,
        random_churn) return 0. SimilarityReactive returns T_init times the number of
        completed clusterings.
        """
