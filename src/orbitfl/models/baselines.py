"""Forecast baselines. All share predict(obs_window, pos_enc) -> (C,) probabilities.

PersistenceForecaster    predicts that the state delta steps ahead equals the last
                         observation.
MarginalForecaster       predicts the per-channel mean occupancy over the window.
EphemerisOnlyForecaster  a small MLP on the positional encoding alone, with no
                         observations. If it matches the CNN, the CNN's skill
                         comes from geography rather than from the observations.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PersistenceForecaster:
    """Predict t+delta = last observation (zero-lag copy)."""

    def predict(
        self,
        obs_window: np.ndarray,
        pos_enc: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        obs_window : (W, C) float32  observed channel states
        pos_enc    : (4,) float32    positional encoding - unused

        Returns
        -------
        np.ndarray  (C,) float32  last observed state (0.0 or 1.0)
        """
        return obs_window[-1].astype(np.float32)


class MarginalForecaster:
    """Running per-channel occupancy mean over the observation window."""

    def predict(
        self,
        obs_window: np.ndarray,
        pos_enc: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        obs_window : (W, C) float32  observed channel states
        pos_enc    : (4,) float32    positional encoding - unused

        Returns
        -------
        np.ndarray  (C,) float32  per-channel empirical occupancy in [0, 1]
        """
        return obs_window.mean(axis=0).astype(np.float32)


class EphemerisOnlyForecaster(nn.Module):
    """Tiny MLP on positional encoding alone; no observation input.

    Architecture: Linear(4, hidden) -> ReLU -> Linear(hidden, n_channels).
    Train with BCE-with-logits; use predict() for numpy inference.
    """

    def __init__(self, n_channels: int, hidden: int = 32) -> None:
        """
        Parameters
        ----------
        n_channels : int  number of spectrum channels C
        hidden     : int  width of hidden layer
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_channels),
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pos : (B, 4) float32  positional encoding

        Returns
        -------
        torch.Tensor  (B, C) logits
        """
        return self.mlp(pos)

    def predict(
        self,
        obs_window: np.ndarray,
        pos_enc: np.ndarray,
    ) -> np.ndarray:
        """
        NumPy inference interface (obs_window is ignored).

        Parameters
        ----------
        obs_window : (W, C) float32  - unused; accepted for interface uniformity
        pos_enc    : (4,) float32    positional encoding

        Returns
        -------
        np.ndarray  (C,) float32  predicted probabilities in [0, 1]
        """
        self.eval()
        with torch.no_grad():
            pos = torch.from_numpy(pos_enc).unsqueeze(0)   # (1, 4)
            logits = self(pos)                               # (1, C)
            return torch.sigmoid(logits).squeeze(0).numpy()  # (C,)
