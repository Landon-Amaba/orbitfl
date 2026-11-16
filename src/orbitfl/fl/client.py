"""Satellite federated learning client.

SatelliteClient holds one satellite's dataset and a private copy of the model.
train_local loads the weights it is given, trains on the samples of the current
round, and returns a ClientUpdate.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from orbitfl.config import FLConfig, ModelConfig
from orbitfl.fl.aggregation import ClientUpdate, StateDict
from orbitfl.models.forecaster import OccupancyForecaster
from orbitfl.world.occupancy import SatDataset


class SatelliteClient:
    """One LEO satellite's local FL participant.

    Each client owns a private copy of the model, so clients do not share state
    between rounds. The loop calls train_local() each round with the weights to
    start from and the slice of the dataset that belongs to the round.
    """

    def __init__(
        self,
        sat_id: int,
        model_state: StateDict,
        dataset: SatDataset,
        cfg: FLConfig,
        model_cfg: ModelConfig,
    ) -> None:
        """
        Parameters
        ----------
        sat_id      : int          LEO satellite id (0 .. N_leo - 1).
        model_state : StateDict    Initial global model weights to load.
        dataset     : SatDataset   This satellite's full observation dataset.
        cfg         : FLConfig     FL hyper-parameters (local_epochs, local_lr, ...).
        model_cfg   : ModelConfig  Forecaster architecture parameters.
        """
        self._sat_id = sat_id
        self._dataset = dataset
        self._cfg = cfg

        self._model = OccupancyForecaster(
            n_channels=dataset._obs.shape[-1],
            window=model_cfg.window,
            use_position=model_cfg.use_position,
            hidden=model_cfg.hidden,
        )
        self._model.load_state_dict(
            {k: v.clone() for k, v in model_state.items()}
        )
        self._use_position = model_cfg.use_position

    # --- Public interface ---

    @property
    def sat_id(self) -> int:
        return self._sat_id

    @property
    def n_dataset_samples(self) -> int:
        return len(self._dataset)

    def train_local(
        self,
        global_state: StateDict,
        round_window: slice,
        device: torch.device | None = None,
        sample_indices: np.ndarray | None = None,
    ) -> ClientUpdate:
        """Load the given weights, train on the selected samples, return the update.

        By default the client trains on the samples in round_window, i.e. those whose
        window ends inside the round's time range. What a client learns in a round
        therefore depends on the region it is flying over.

        Parameters
        ----------
        global_state   : StateDict        Model weights to start from.
        round_window   : slice            Sample indices [start, stop) for this round.
        device         : torch.device     Compute device; CPU if None.
        sample_indices : np.ndarray|None  Explicit dataset indices to train on. When
            given, overrides round_window. Used by cluster_local_train to restrict a
            client to the samples collected while it belonged to its current cluster.

        Returns
        -------
        ClientUpdate  Post-training state_dict plus accounting metadata.
        """
        if device is None:
            device = torch.device("cpu")

        # Load the starting weights
        self._model.load_state_dict(
            {k: v.clone() for k, v in global_state.items()}
        )
        self._model.to(device)

        # Select training samples: explicit indices if given, else the round window.
        if sample_indices is not None:
            indices = [int(i) for i in sample_indices]
            stop = int(sample_indices.max()) + 1 if len(sample_indices) else 0
        else:
            start, stop, _ = round_window.indices(len(self._dataset))
            indices = list(range(start, stop))

        if not indices:
            # No data to train on: return the starting weights unchanged
            return ClientUpdate(
                state_dict={k: v.detach().clone().cpu() for k, v in global_state.items()},
                n_samples=0,
                grad_norm=0.0,
                sat_id=self._sat_id,
                computed_at_t=stop,
            )

        subset = Subset(self._dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=min(self._cfg.local_batch, len(indices)),
            shuffle=True,
            num_workers=0,
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._cfg.local_lr
        )

        self._model.train()
        total_grad_norm = 0.0

        for _ in range(self._cfg.local_epochs):
            for obs_win, pos_enc, target in loader:
                obs_win = obs_win.to(device)
                pos_enc = pos_enc.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                logits = self._model(
                    obs_win, pos_enc if self._use_position else None
                )
                loss = criterion(logits, target)
                loss.backward()
                total_grad_norm += _grad_norm(self._model)
                optimizer.step()

        return ClientUpdate(
            state_dict={
                k: v.detach().clone().cpu()
                for k, v in self._model.state_dict().items()
            },
            n_samples=len(indices),
            grad_norm=float(total_grad_norm),
            sat_id=self._sat_id,
            computed_at_t=stop,
        )

    def current_state_dict(self) -> StateDict:
        """Return a copy of the client's current model parameters."""
        return {k: v.detach().clone() for k, v in self._model.state_dict().items()}


# --- Helpers ---


def _grad_norm(model: nn.Module) -> float:
    """Sum of L2 norms of all non-None parameter gradients."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.norm(2).item())
    return total
