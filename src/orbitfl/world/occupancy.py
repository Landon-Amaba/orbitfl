"""Ground-truth occupancy field and satellite observation streams.

OccupancyWorld simulates one two-state Markov chain per region and channel at
construction and keeps the result in memory as an (R, T, C) uint8 array, where
R is the number of regions, T the number of timesteps and C the number of
channels. Per-channel transition probabilities:
    p(busy -> idle) = dt / mean_on_s[c]   (clamped to [0, 1])
    p(idle -> busy) = dt / mean_off_s[c]  (clamped to [0, 1])
The initial state is Bernoulli(occupancy[c]).

Main methods
    truth(region_id, t_s)             (C,) ground-truth occupancy in {0, 1}
    observe(lat, lon, t_s, rng)       (C,) truth at the region plus symmetric flip noise
    distribution_params(lat, lon)     RegionProfile at a point
    generate_obs_sequence(...)        (T, C) observations along a lat/lon track
    observation_stream(trace, ...)    SatDataset for one satellite of a trace

SatDataset is a torch Dataset of (obs_window (W, C), pos_enc (4,), target (C,))
triples, where pos_enc = [sin(lat), cos(lat), sin(lon), cos(lon)] at the end of
the window. Observations are generated once when the dataset is built. World
parameters can therefore be changed without regenerating the trace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from orbitfl.config import WorldConfig
from orbitfl.world.regions import (
    RegionMap,
    RegionProfile,
    build_region_map,
    concept_shift_channel_mask,
    contrast_dwell_times,
    region_at,
    region_polarity,
)

if TYPE_CHECKING:
    from orbitfl.trace.reader import Trace


# --- SatDataset ---


class SatDataset:
    """Torch Dataset of (obs_window, pos_enc, target) triples for one satellite.

      obs_window : torch.Tensor (W, C) float32  W consecutive noisy observations
      pos_enc    : torch.Tensor (4,)   float32  positional encoding at window end:
                   [sin(lat_rad), cos(lat_rad), sin(lon_rad), cos(lon_rad)]
      target     : torch.Tensor (C,)   float32  observation delta_steps after window end

    Length: max(0, T - window - delta_steps + 1) valid windows.
    """

    def __init__(
        self,
        obs: np.ndarray,        # (T, C) float32 - pre-generated observations
        lat: np.ndarray,  # (T,) float64, sub-satellite latitudes in degrees
        lon: np.ndarray,  # (T,) float64, sub-satellite longitudes in degrees
        window: int,
        delta_steps: int,
        target_obs: np.ndarray | None = None,  # (T, C) float32 - target stream; None -> obs
    ) -> None:
        """Parameters
        ----------
        obs         : (T, C) float32  All T observations for one satellite (input windows).
        lat         : (T,)            Sub-satellite latitudes in degrees.
        lon         : (T,)            Sub-satellite longitudes in degrees.
        window      : int             History window length W (cfg.model.window).
        delta_steps : int             Forecast horizon in timesteps.
        target_obs  : (T, C) float32 or None  Stream the delta-ahead target is read from.
                      None means the target comes from obs itself. With concept shift
                      enabled, a polarity-inverted stream is passed here, so the input
                      window is unchanged while the label depends on the region (see
                      OccupancyWorld._apply_concept_shift).
        """
        import torch

        T = obs.shape[0]
        self._window = window
        self._delta = delta_steps
        self._n = max(0, T - window - delta_steps + 1)

        # Store observations as a contiguous float32 tensor: (T, C)
        self._obs: torch.Tensor = torch.from_numpy(obs)
        # Target stream: the same buffer as _obs unless a separate one was supplied.
        self._target_obs: torch.Tensor = (
            self._obs if target_obs is None else torch.from_numpy(target_obs)
        )

        # Pre-compute positional encodings for all timesteps: (T, 4)
        lat_rad = np.radians(lat).astype(np.float32)
        lon_rad = np.radians(lon).astype(np.float32)
        pos_arr = np.stack(
            [np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)],
            axis=-1,
        )  # (T, 4) float32
        self._pos_enc: torch.Tensor = torch.from_numpy(pos_arr)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int):
        """Return the i-th (obs_window, pos_enc, target) triple.

        Returns
        -------
        obs_window : torch.Tensor (W, C) float32
        pos_enc    : torch.Tensor (4,)   float32  positional encoding at the window-end step
        target     : torch.Tensor (C,)   float32
        """
        # Window spans [i, i+window); window-end index is i+window-1
        end = i + self._window - 1
        obs_window = self._obs[i : i + self._window]          # (W, C)
        pos_enc = self._pos_enc[end]                           # (4,)
        target = self._target_obs[end + self._delta]           # (C,)
        return obs_window, pos_enc, target


class OccupancyWorld:
    """Per-region two-state Markov chain world model.

    The ground-truth chains are drawn from one Generator, consumed in a fixed order
    in __init__, so a seed fully determines them. Observation noise is drawn from
    separate Generators, which keeps the ground truth independent of how many
    observations are generated.
    """

    def __init__(
        self,
        cfg: WorldConfig,
        seed: int,
        timestep_s: int = 15,
        n_steps: int = 5760,
    ) -> None:
        """
        Parameters
        ----------
        cfg        : WorldConfig
        seed       : int    master seed for the ground-truth chains
        timestep_s : int    Markov chain step duration in seconds
        n_steps    : int    number of timesteps to simulate
        """
        self._C = cfg.n_channels
        self._dt = float(timestep_s)
        self._T = n_steps
        self._flip_noise = cfg.flip_noise
        self._seed = seed  # used to seed observation_stream

        self._region_map: RegionMap = build_region_map(cfg)
        self._R = len(self._region_map)
        # Inter-region predictability contrast (0 = same for all regions; see regions.py).
        self._contrast = float(getattr(cfg, "predictability_contrast", 0.0))
        # Hidden concept-shift strength (0 = no target inversion; see regions.py).
        self._concept_shift = float(getattr(cfg, "concept_shift", 0.0))

        # chains: (R, T, C) uint8, ground-truth occupancy per region and timestep
        self._chains = self._simulate_all(np.random.default_rng(seed))

    # --- Public interface ---

    def truth(self, region_id: int, t_s: float) -> np.ndarray:
        """
        Ground-truth channel states for a region at time t_s.

        Parameters
        ----------
        region_id : int    index into the region map
        t_s       : float  seconds since epoch

        Returns
        -------
        np.ndarray  (C,) float32  occupancy in {0.0, 1.0}
        """
        t_idx = min(int(t_s / self._dt), self._T - 1)
        return self._chains[region_id, t_idx].astype(np.float32)

    def observe(
        self,
        lat: float,
        lon: float,
        t_s: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Noisy observation of channel states at (lat, lon, t_s).

        Each channel is independently flipped with probability cfg.flip_noise
        (symmetric noise model).

        Parameters
        ----------
        lat : float  sub-satellite latitude  in degrees
        lon : float  sub-satellite longitude in degrees
        t_s : float  seconds since epoch
        rng : np.random.Generator  caller-owned; consumed for flip noise only

        Returns
        -------
        np.ndarray  (C,) float32  observed channel states in {0.0, 1.0}
        """
        r = region_at(lat, lon, self._region_map)
        t_idx = min(int(t_s / self._dt), self._T - 1)
        truth = self._chains[r, t_idx]                            # (C,) uint8
        flip = (rng.random(self._C) < self._flip_noise).astype(np.uint8)  # (C,)
        return (truth ^ flip).astype(np.float32)                  # (C,)

    def distribution_params(self, lat: float, lon: float) -> RegionProfile:
        """Return the RegionProfile for the region containing (lat, lon).

        The result depends only on (lat, lon): a satellite's data distribution is a
        known function of its position.

        Parameters
        ----------
        lat : float  sub-satellite latitude  in degrees
        lon : float  sub-satellite longitude in degrees

        Returns
        -------
        RegionProfile
        """
        return self._region_map[region_at(lat, lon, self._region_map)]

    def generate_obs_sequence(
        self,
        lat_seq: np.ndarray,
        lon_seq: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Observation stream along one satellite trajectory.

        Equivalent to calling observe() at every timestep, without the Python loop.
        Takes lat/lon arrays directly, so it can be used without a trace.

        Parameters
        ----------
        lat_seq : np.ndarray  (T,) sub-satellite latitudes  in degrees
        lon_seq : np.ndarray  (T,) sub-satellite longitudes in degrees
        rng     : np.random.Generator  caller-owned

        Returns
        -------
        np.ndarray  (T, C) float32  observed channel states in {0.0, 1.0}
        """
        T = len(lat_seq)
        region_ids = np.array(
            [region_at(float(lat_seq[t]), float(lon_seq[t]), self._region_map)
             for t in range(T)],
            dtype=np.intp,
        )
        t_indices = np.minimum(np.arange(T, dtype=np.intp), self._T - 1)
        truth = self._chains[region_ids, t_indices]                      # (T, C) uint8
        flip = (rng.random((T, self._C)) < self._flip_noise).astype(np.uint8)  # (T, C)
        return (truth ^ flip).astype(np.float32)                         # (T, C)

    def observation_stream(
        self,
        trace: "Trace",
        sat_id: int,
        window: int = 16,
        delta_steps: int = 4,
    ) -> SatDataset:
        """Build the SatDataset for one satellite of a trace.

        The satellite's sub-satellite track is replayed through the cached ground-truth
        chains and flip noise is applied. This happens once; the trace is not read
        again afterwards.

        The noise generator is seeded from (world_seed, sat_id), so the same trace and
        world always give the same dataset.

        Parameters
        ----------
        trace       : Trace   Trace loaded via Trace.load().
        sat_id      : int     LEO satellite id (0 .. N_leo - 1).
        window      : int     History window W in timesteps (default cfg.model.window).
        delta_steps : int     Forecast horizon in timesteps (default cfg.experiment.delta_steps).

        Returns
        -------
        SatDataset  Torch dataset of (obs_window (W, C), pos_enc (4,), target (C,)) triples.
        """
        subpts = trace.subpoints(sat_id)   # (T_trace, 2)  lat, lon
        lat: np.ndarray = subpts[:, 0]    # (T_trace,) degrees
        lon: np.ndarray = subpts[:, 1]    # (T_trace,) degrees

        # Flip-noise generator, seeded from (world_seed, sat_id)
        rng = np.random.default_rng([self._seed, sat_id])
        obs = self.generate_obs_sequence(lat, lon, rng)  # (T_trace, C) float32

        # Concept shift: build a separate target stream whose conflicting channels are
        # complemented in regions of polarity -1. With concept_shift == 0 the target is
        # the observation stream itself.
        region_ids = np.array(
            [region_at(float(lat[t]), float(lon[t]), self._region_map)
             for t in range(len(lat))],
            dtype=np.intp,
        )  # (T_trace,)
        target_obs = self._apply_concept_shift(obs, region_ids)  # (T_trace, C) or same obj

        return SatDataset(
            obs, lat, lon, window=window, delta_steps=delta_steps,
            target_obs=None if target_obs is obs else target_obs,
        )

    def _apply_concept_shift(
        self, obs: np.ndarray, region_ids: np.ndarray
    ) -> np.ndarray:
        """Build the delta-ahead target stream with the per-region polarity flip.

        For every timestep whose region has polarity -1 (region_polarity), the
        conflicting channels (concept_shift_channel_mask) are complemented (1 - value).
        The observation window is untouched, so a model cannot tell from its input
        whether the future on those channels is persistent or anti-persistent; only the
        region decides.

        Returns the same obs object, not a copy, when concept_shift == 0.

        Parameters
        ----------
        obs        : (T, C) float32  Input observation stream.
        region_ids : (T,)   intp     Region index at each timestep.

        Returns
        -------
        np.ndarray  (T, C) float32  Target stream (obs itself if concept_shift == 0).
        """
        mask = concept_shift_channel_mask(self._C, self._concept_shift)  # (C,) bool
        if not mask.any():
            return obs
        # (T,) bool, timesteps whose region has inverted targets.
        polarity_neg = np.array(
            [region_polarity(int(r)) < 0 for r in region_ids], dtype=bool
        )
        target_obs = obs.copy()
        flip = np.outer(polarity_neg, mask)          # (T, C) bool - cells to complement
        target_obs[flip] = 1.0 - target_obs[flip]
        return target_obs

    # --- Internal ---

    def _simulate_region(
        self, rp: RegionProfile, rng: np.random.Generator, region_idx: int = 0
    ) -> np.ndarray:
        """
        Simulate one region's two-state Markov chain.

        Parameters
        ----------
        rp         : RegionProfile
        rng        : np.random.Generator
        region_idx : int  Region rank, used only when predictability_contrast > 0.

        Returns
        -------
        np.ndarray  (T, C) uint8
        """
        # Spread this region's predictability relative to the others, holding occupancy
        # fixed. Has no effect when contrast == 0.
        mean_on_s, mean_off_s = contrast_dwell_times(
            rp.mean_on_s, rp.mean_off_s, self._dt, region_idx, self._R, self._contrast
        )
        p_on = np.clip(self._dt / mean_on_s, 0.0, 1.0)    # (C,) P(busy -> idle)
        p_off = np.clip(self._dt / mean_off_s, 0.0, 1.0)   # (C,) P(idle -> busy)

        # Initial state: Bernoulli(occupancy[c]) per channel
        state = (rng.random(self._C) < rp.occupancy).astype(np.uint8)  # (C,)

        chain = np.empty((self._T, self._C), dtype=np.uint8)
        chain[0] = state

        for t in range(1, self._T):
            # Transition probability is p_on where busy, p_off where idle
            p_flip = np.where(state, p_on, p_off)              # (C,)
            flip = (rng.random(self._C) < p_flip).astype(np.uint8)  # (C,)
            state = state ^ flip                                # (C,)
            chain[t] = state

        return chain

    def _simulate_all(self, rng: np.random.Generator) -> np.ndarray:
        """
        Simulate all R regions sequentially, sharing one Generator.

        Returns
        -------
        np.ndarray  (R, T, C) uint8
        """
        return np.stack(
            [self._simulate_region(rp, rng, k) for k, rp in enumerate(self._region_map)],
            axis=0,
        )
