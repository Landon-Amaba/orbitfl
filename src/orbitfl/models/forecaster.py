"""Occupancy forecaster network.

positional_encoding(lat_deg, lon_deg) returns [sin(lat), cos(lat), sin(lon), cos(lon)].

OccupancyForecaster is a small CNN: three Conv1d layers over the time axis
(spectrum channels as feature maps) followed by two fully connected layers that
output one logit per channel. The use_position flag controls whether the
positional encoding is fed to the fully connected head; turning it off hides the
satellite's position from the model.

The observations at lags 0, 4, 8 and 12 are concatenated directly into the
fully connected input next to the convolutional features. The convolutions
(receptive field 5) smooth the recent history, and the lag inputs give the head
access to the whole 16-step window without dilated convolutions, which are very
slow on CPU in PyTorch.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def positional_encoding(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Map a lat/lon point to a 4-vector that varies continuously with position.

    [sin(lat), cos(lat), sin(lon), cos(lon)] has no jump at the date line or at the
    poles.

    Parameters
    ----------
    lat_deg : float  sub-satellite latitude  in degrees (-90 ... 90)
    lon_deg : float  sub-satellite longitude in degrees (-180 ... 180)

    Returns
    -------
    np.ndarray  (4,) float32  [sin_lat, cos_lat, sin_lon, cos_lon]
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    return np.array(
        [math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)],
        dtype=np.float32,
    )


class OccupancyForecaster(nn.Module):
    """Three Conv1d layers over the time axis plus two FC layers; returns logits.

    Data flow:
      obs (B, W, C) -> permute -> (B, C, W) -> Conv1d x 3 -> (B, hidden, W)
      -> x[:, :, -1] -> (B, hidden)   conv features at the last timestep
      -> cat obs[:,-1,:]  (B, C)      lag 0
      -> cat obs[:,-5,:]  (B, C)      lag 4
      -> cat obs[:,-9,:]  (B, C)      lag 8
      -> cat obs[:,-13,:] (B, C)      lag 12
      -> [cat pos (B, 4) if use_position]
      -> (B, hidden + 4*C [+ 4]) -> Linear -> ReLU -> Linear -> (B, C) logits

    Apply torch.sigmoid to the logits to get per-channel probabilities.
    """

    def __init__(
        self,
        n_channels: int,
        window: int,
        use_position: bool,
        hidden: int = 64,
    ) -> None:
        """
        Parameters
        ----------
        n_channels   : int   number of spectrum channels C
        window       : int   observation window length W (informational only)
        use_position : bool  whether to concatenate positional encoding before FC head
        hidden       : int   width of conv and FC layers
        """
        super().__init__()
        self._use_position = use_position
        self._n_channels = n_channels
        self._window = window  # stored for documentation; not used in forward

        # 1-D conv over time dimension; input (B, C, W) -> (B, hidden, W)
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # No pooling: forward takes the last time position

        # FC head: (hidden + 4*C [+ 4]) -> hidden -> n_channels
        # hidden:  conv features at the last timestep
        # 4*C:     observations at lags 0, 4, 8, 12
        # 4:       positional encoding (if use_position)
        fc_in = hidden + 4 * n_channels + (4 if use_position else 0)
        self.fc = nn.Sequential(
            nn.Linear(fc_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_channels),
        )

    def forward(
        self,
        obs: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : (B, W, C) float32  observed channel states over window
        pos : (B, 4) float32 or None  positional encoding; ignored when use_position=False

        Returns
        -------
        torch.Tensor  (B, C) logits  (apply sigmoid for probabilities)
        """
        # (B, W, C) -> (B, C, W) for Conv1d
        x = obs.permute(0, 2, 1)          # (B, C, W)
        x = self.conv(x)                   # (B, hidden, W)
        x = x[:, :, -1]  # (B, hidden), conv features at the last timestep

        # Observations at lags 0, 4, 8, 12, clamped to the window so that W < 13 still works
        W = obs.shape[1]
        lag0  = obs[:, -1,              :]   # (B, C)
        lag4  = obs[:, max(0, W - 5),   :]   # (B, C)
        lag8  = obs[:, max(0, W - 9),   :]   # (B, C)
        lag12 = obs[:, max(0, W - 13),  :]   # (B, C)
        x = torch.cat([x, lag0, lag4, lag8, lag12], dim=-1)   # (B, hidden + 4*C)

        if self._use_position and pos is not None:
            x = torch.cat([x, pos], dim=-1)    # (B, hidden + 4*C + 4)

        return self.fc(x)                  # (B, C)
