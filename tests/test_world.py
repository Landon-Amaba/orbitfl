"""Tests for src/orbitfl/world/.

W1  distribution_params(lat, lon) returns the same profile on repeated calls
    and across OccupancyWorld instances with the same seed.
W2  The measured per-channel occupancy of a long chain is close to the
    configured stationary rate (within 0.05 over at least 50 000 steps at 15 s).
W3  The fraction of observe() results that differ from the truth is close to
    flip_noise (within 0.01 over 10 000 samples).
W4  observation_stream on a trace: SatDataset shapes, types and determinism.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from orbitfl.config import RegionConfig, WorldConfig
from orbitfl.world.occupancy import OccupancyWorld, SatDataset
from orbitfl.world.regions import (
    concept_shift_channel_mask,
    contrast_dwell_times,
    region_at,
    region_polarity,
)

# --- Shared minimal world config (2 regions - no real lat/lon boxes needed here) ---

def _two_region_cfg(n_channels: int = 8, flip_noise: float = 0.05) -> WorldConfig:
    """WorldConfig with one busy region (lat > 0) and one quiet region (catch-all)."""
    return WorldConfig(
        n_channels=n_channels,
        flip_noise=flip_noise,
        beam_half_cone_deg=20.0,
        regions=[
            RegionConfig(
                name="busy",
                lat_min=0.0,
                lat_max=90.0,
                lon_min=-180.0,
                lon_max=180.0,
                occupancy=[0.8],
                mean_on_s=[20.0],
                mean_off_s=[5.0],
            ),
            RegionConfig(
                name="quiet",
                lat_min=-90.0,
                lat_max=0.0,
                lon_min=-180.0,
                lon_max=180.0,
                occupancy=[0.1],
                mean_on_s=[10.0],
                mean_off_s=[90.0],
            ),
        ],
    )


# --- W1 - distribution_params is deterministic ---

class TestDistributionParamsDeterminism:
    """distribution_params(lat, lon) must be deterministic in lat/lon (the thesis premise)."""

    def test_repeated_calls_same_object(self):
        """Same world, same point -> identical RegionProfile (field-wise)."""
        cfg = _two_region_cfg()
        world = OccupancyWorld(cfg, seed=0)
        p1 = world.distribution_params(45.0, 10.0)
        p2 = world.distribution_params(45.0, 10.0)
        assert p1 == p2, "distribution_params not reproducible on repeated calls"

    def test_separate_instances_same_seed(self):
        """Two OccupancyWorld instances, same seed, same point -> same profile."""
        cfg = _two_region_cfg()
        w1 = OccupancyWorld(cfg, seed=7)
        w2 = OccupancyWorld(cfg, seed=99)   # different seed - must not matter
        # The profile is determined purely by lat/lon, not by the chain seed
        p1 = w1.distribution_params(45.0, 10.0)
        p2 = w2.distribution_params(45.0, 10.0)
        assert p1 == p2, "distribution_params varies with world seed (should not)"

    def test_different_regions_differ(self):
        """Points in different regions return different profiles."""
        cfg = _two_region_cfg()
        world = OccupancyWorld(cfg, seed=0)
        p_busy = world.distribution_params(45.0, 0.0)    # lat > 0 -> busy
        p_quiet = world.distribution_params(-45.0, 0.0)  # lat < 0 -> quiet
        assert p_busy.name == "busy"
        assert p_quiet.name == "quiet"
        assert not np.array_equal(p_busy.occupancy, p_quiet.occupancy)

    def test_returns_region_profile(self):
        """Return type is a RegionProfile with correct array shape."""
        from orbitfl.world.regions import RegionProfile
        cfg = _two_region_cfg(n_channels=16)
        world = OccupancyWorld(cfg, seed=0)
        rp = world.distribution_params(10.0, 0.0)
        assert isinstance(rp, RegionProfile)
        assert rp.occupancy.shape == (16,)
        assert rp.mean_on_s.shape == (16,)
        assert rp.mean_off_s.shape == (16,)


# --- W2 - empirical occupancy ~ configured stationary rate ---

class TestEmpiricalOccupancy:
    """
    Long Markov chain must converge to configured stationary occupancy.

    Stationary distribution of a two-state chain: pi_1 = p_off / (p_on + p_off)
    where p_on = dt/mean_on_s, p_off = dt/mean_off_s (both clamped to 1).
    The test uses mean_on_s = 200 s so that dt = 15 s gives p_on ~ 0.075,
    which keeps both transition probabilities well below 1 (no clamping).
    """

    @pytest.mark.parametrize("occupancy,tol", [(0.8, 0.05), (0.1, 0.05)])
    def test_chain_converges_to_stationary(self, occupancy: float, tol: float):
        """Empirical mean of a long chain ~ stationary occupancy."""
        # mean_on_s = 200 s, mean_off_s chosen so pi_1 = occupancy:
        #   pi_1 = (dt/mean_off) / (dt/mean_on + dt/mean_off)
        #        = mean_on / (mean_on + mean_off) = occupancy
        #   -> mean_off = mean_on * (1 - occupancy) / occupancy
        mean_on = 200.0
        mean_off = mean_on * (1.0 - occupancy) / max(occupancy, 1e-9)
        cfg = WorldConfig(
            n_channels=4,
            flip_noise=0.0,  # no noise so truth = observation exactly
            beam_half_cone_deg=20.0,
            regions=[
                RegionConfig(
                    name="test",
                    lat_min=-90.0,
                    lat_max=90.0,
                    lon_min=-180.0,
                    lon_max=180.0,
                    occupancy=[occupancy],
                    mean_on_s=[mean_on],
                    mean_off_s=[mean_off],
                ),
            ],
        )
        # 50 000 steps at dt = 15 s -> 208 h of simulated chain time
        world = OccupancyWorld(cfg, seed=0, timestep_s=15, n_steps=50_000)
        chain_mean = world._chains[0].mean()  # mean over (T, C)
        assert abs(chain_mean - occupancy) < tol, (
            f"Chain mean {chain_mean:.4f} deviates from target {occupancy:.2f} "
            f"by more than ±{tol}"
        )

    def test_busy_region_higher_than_quiet(self):
        """Busy region chain mean > quiet region chain mean."""
        cfg = _two_region_cfg()
        world = OccupancyWorld(cfg, seed=42, n_steps=20_000)
        busy_mean = world._chains[0].mean()
        quiet_mean = world._chains[1].mean()
        assert busy_mean > quiet_mean, (
            f"Busy mean {busy_mean:.3f} not above quiet mean {quiet_mean:.3f}"
        )


# --- W3 - flip noise at configured rate ---

class TestFlipNoise:
    """observe() flips each channel independently with probability flip_noise."""

    def test_flip_rate_approx_configured(self):
        """
        Over 10 000 single-timestep observations at t=0, fraction of channels
        that differ from truth ~ flip_noise (within +/-0.01).
        """
        flip_noise = 0.10
        C = 16
        N_SAMPLES = 10_000

        cfg = WorldConfig(
            n_channels=C,
            flip_noise=flip_noise,
            beam_half_cone_deg=20.0,
            regions=[
                RegionConfig(
                    name="all",
                    lat_min=-90.0,
                    lat_max=90.0,
                    lon_min=-180.0,
                    lon_max=180.0,
                    occupancy=[0.5],
                    mean_on_s=[30.0],
                    mean_off_s=[30.0],
                ),
            ],
        )
        world = OccupancyWorld(cfg, seed=0, n_steps=100)
        truth = world.truth(0, t_s=0.0)  # (C,) ground truth at t=0

        rng = np.random.default_rng(1)
        total_flips = 0
        for _ in range(N_SAMPLES):
            obs = world.observe(0.0, 0.0, t_s=0.0, rng=rng)  # (C,)
            total_flips += int(np.sum(obs != truth))

        empirical_rate = total_flips / (N_SAMPLES * C)
        assert abs(empirical_rate - flip_noise) < 0.01, (
            f"Empirical flip rate {empirical_rate:.4f} deviates from "
            f"configured {flip_noise} by more than ±0.01"
        )

    def test_zero_flip_noise_matches_truth(self):
        """With flip_noise=0, observe() always equals truth."""
        cfg = WorldConfig(
            n_channels=8,
            flip_noise=0.0,
            beam_half_cone_deg=20.0,
            regions=[
                RegionConfig(
                    name="all",
                    lat_min=-90.0,
                    lat_max=90.0,
                    lon_min=-180.0,
                    lon_max=180.0,
                    occupancy=[0.5],
                    mean_on_s=[30.0],
                    mean_off_s=[30.0],
                ),
            ],
        )
        world = OccupancyWorld(cfg, seed=0)
        rng = np.random.default_rng(0)
        for t in range(0, 300, 15):
            truth = world.truth(0, float(t))
            obs = world.observe(0.0, 0.0, float(t), rng)
            np.testing.assert_array_equal(obs, truth, err_msg=f"Mismatch at t={t}")

    def test_generate_obs_sequence_shape_and_range(self):
        """generate_obs_sequence returns (T, C) float32 with values in {0, 1}."""
        C = 8
        T = 100
        cfg = _two_region_cfg(n_channels=C)
        world = OccupancyWorld(cfg, seed=0, n_steps=T)
        lat_seq = np.full(T, 45.0)   # always busy region
        lon_seq = np.zeros(T)
        rng = np.random.default_rng(0)
        obs = world.generate_obs_sequence(lat_seq, lon_seq, rng)
        assert obs.shape == (T, C), f"Expected ({T}, {C}), got {obs.shape}"
        assert obs.dtype == np.float32
        assert np.all((obs == 0.0) | (obs == 1.0)), "Values must be binary {0, 1}"


# --- W4 - observation_stream wired to trace ---

_GOLDEN_DIR = Path(__file__).parent / "golden" / "tiny_trace"
_TINY_YAML = Path(__file__).parent.parent / "configs" / "experiments" / "tiny_trace.yaml"
_BASE_YAML = Path(__file__).parent.parent / "configs" / "base.yaml"


def _tiny_world(n_steps: int = 240) -> OccupancyWorld:
    """Minimal world matching the tiny golden trace (T=240, C=1)."""
    return OccupancyWorld(_two_region_cfg(n_channels=1), seed=0,
                          timestep_s=15, n_steps=n_steps)


class TestObservationStream:
    """observation_stream on the tiny golden trace."""

    @pytest.fixture(autouse=True)
    def require_golden(self):
        if not _GOLDEN_DIR.exists():
            pytest.skip("Golden trace not found; run make_trace.py first")

    def _load_trace(self):
        from orbitfl.trace.reader import Trace
        return Trace.load(_GOLDEN_DIR)

    def test_returns_sat_dataset(self):
        """observation_stream returns a SatDataset instance."""
        trace = self._load_trace()
        world = _tiny_world()
        ds = world.observation_stream(trace, sat_id=0, window=4, delta_steps=2)
        assert isinstance(ds, SatDataset)

    def test_dataset_length(self):
        """SatDataset length = T - window - delta_steps + 1."""
        W, D = 4, 2
        trace = self._load_trace()
        T = trace.n_timesteps   # 240 for the tiny trace
        world = _tiny_world(n_steps=T)
        ds = world.observation_stream(trace, sat_id=0, window=W, delta_steps=D)
        assert len(ds) == T - W - D + 1

    def test_item_shapes(self):
        """Each item has correct shapes and dtypes."""
        W, C, D = 4, 1, 2
        trace = self._load_trace()
        world = _tiny_world()
        ds = world.observation_stream(trace, sat_id=0, window=W, delta_steps=D)
        obs_win, pos_enc, target = ds[0]
        assert obs_win.shape == (W, C), f"obs_window shape: {obs_win.shape}"
        assert pos_enc.shape == (4,), f"pos_enc shape: {pos_enc.shape}"
        assert target.shape == (C,), f"target shape: {target.shape}"
        assert obs_win.dtype == torch.float32
        assert pos_enc.dtype == torch.float32
        assert target.dtype == torch.float32

    def test_obs_values_binary(self):
        """obs_window and target values are binary {0.0, 1.0}."""
        trace = self._load_trace()
        world = _tiny_world()
        ds = world.observation_stream(trace, sat_id=0, window=4, delta_steps=2)
        for i in range(min(20, len(ds))):
            obs_win, _, target = ds[i]
            assert torch.all((obs_win == 0.0) | (obs_win == 1.0))
            assert torch.all((target == 0.0) | (target == 1.0))

    def test_pos_enc_in_range(self):
        """Positional encoding values are in [-1, 1] (sin/cos)."""
        trace = self._load_trace()
        world = _tiny_world()
        ds = world.observation_stream(trace, sat_id=0, window=4, delta_steps=2)
        _, pos_enc, _ = ds[0]
        assert torch.all(pos_enc >= -1.0) and torch.all(pos_enc <= 1.0)

    def test_deterministic_across_calls(self):
        """Two observation_stream calls with the same (sat_id, seed) are identical."""
        trace = self._load_trace()
        world = _tiny_world()
        ds1 = world.observation_stream(trace, sat_id=1, window=4, delta_steps=2)
        ds2 = world.observation_stream(trace, sat_id=1, window=4, delta_steps=2)
        obs1, _, t1 = ds1[5]
        obs2, _, t2 = ds2[5]
        assert torch.equal(obs1, obs2), "observation_stream is not reproducible"
        assert torch.equal(t1, t2)

    def test_different_sats_differ(self):
        """Two satellites at different positions produce different observations."""
        trace = self._load_trace()
        N_leo = trace.n_leo
        if N_leo < 2:
            pytest.skip("Need at least 2 LEOs")
        world = _tiny_world()
        ds0 = world.observation_stream(trace, sat_id=0, window=4, delta_steps=2)
        ds1 = world.observation_stream(trace, sat_id=1, window=4, delta_steps=2)
        # Collect 10 items from each and check they differ at some point
        obs_0 = torch.stack([ds0[i][0] for i in range(10)])   # (10, W, C)
        obs_1 = torch.stack([ds1[i][0] for i in range(10)])   # (10, W, C)
        assert not torch.equal(obs_0, obs_1), (
            "Different satellites produced identical observation streams; "
            "check that subpoints are actually different"
        )


class TestPredictabilityContrast:
    """contrast_dwell_times spreads per-region r4 while preserving occupancy."""

    def test_zero_contrast_is_identity(self):
        """contrast == 0 must return the dwell times unchanged (default world intact)."""
        on = np.array([240.0])
        off = np.array([100.0])
        on2, off2 = contrast_dwell_times(on, off, dt=15.0, region_idx=2, n_regions=4,
                                         contrast=0.0)
        assert np.array_equal(on2, on) and np.array_equal(off2, off)

    def test_contrast_preserves_occupancy_and_spreads_r4(self):
        """Occupancy pi is invariant; r4 decreases monotonically from region 0 to last."""
        dt, n, c = 15.0, 4, 0.8
        on = np.array([240.0])
        off = np.array([100.0])
        p_sum0 = dt / on + dt / off
        pi0 = float((dt / off / p_sum0)[0])  # target stationary occupancy

        r4s, occs = [], []
        for k in range(n):
            on_k, off_k = contrast_dwell_times(on, off, dt, k, n, c)
            p_on, p_off = dt / on_k, dt / off_k
            p_sum = p_on + p_off
            occs.append(float((p_off / p_sum)[0]))
            r4s.append(float((1.0 - p_sum)[0] ** 4))

        assert all(abs(o - pi0) < 1e-9 for o in occs), "occupancy not preserved"
        assert r4s[0] > r4s[-1], "region 0 should be more predictable than the last"
        assert all(r4s[i] >= r4s[i + 1] for i in range(n - 1)), "r4 not monotone by rank"


# --- Hidden concept shift ---


def _polarity_two_region_cfg(
    n_channels: int, concept_shift: float
) -> WorldConfig:
    """Two regions with the same dynamics and opposite concept-shift polarity.

    Region 0 (lat > 0, even rank) has polarity +1 and region 1 (lat < 0, odd rank)
    has polarity -1. Both have occupancy 0.5 and symmetric 150 s dwell times
    (p_on = p_off = 0.1, r4 ~ 0.41), so their input windows are statistically the
    same and any difference in the target between regions is hidden from the input.
    """
    return WorldConfig(
        n_channels=n_channels,
        flip_noise=0.05,
        beam_half_cone_deg=20.0,
        concept_shift=concept_shift,
        regions=[
            RegionConfig(name="pos", lat_min=0.0, lat_max=90.0, lon_min=-180.0,
                         lon_max=180.0, occupancy=[0.5], mean_on_s=[150.0],
                         mean_off_s=[150.0]),
            RegionConfig(name="neg", lat_min=-90.0, lat_max=0.0, lon_min=-180.0,
                         lon_max=180.0, occupancy=[0.5], mean_on_s=[150.0],
                         mean_off_s=[150.0]),
        ],
    )


def _persistence_accuracy(
    obs: np.ndarray, target: np.ndarray, delta: int
) -> float:
    """Accuracy of the naive 'predict last observation' rule against `target`."""
    # obs[t] used to predict target[t + delta]; averaged over t and channels.
    return float((obs[: -delta] == target[delta:]).mean())


def _lag1_autocorr(seq: np.ndarray) -> float:
    """Mean lag-1 autocorrelation of a binary (T, C) stream over channels."""
    x = seq - seq.mean(axis=0, keepdims=True)
    num = (x[:-1] * x[1:]).sum(axis=0)
    den = (x * x).sum(axis=0)
    return float(np.mean(num / np.where(den == 0, 1.0, den)))


class TestConceptShiftHelpers:
    """Pure helpers region_polarity / concept_shift_channel_mask."""

    def test_polarity_splits_by_rank_parity(self):
        assert [region_polarity(i) for i in range(4)] == [1, -1, 1, -1]

    def test_zero_concept_shift_masks_nothing(self):
        mask = concept_shift_channel_mask(32, 0.0)
        assert mask.dtype == bool and mask.shape == (32,) and not mask.any()

    def test_channel_mask_grows_monotonically(self):
        counts = [int(concept_shift_channel_mask(32, c).sum())
                  for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert counts == [0, 8, 16, 24, 32], f"non-monotone mask counts {counts}"


class TestConceptShiftWorld:
    """concept_shift = 0 leaves the world unchanged; concept_shift > 0 creates a
    target conflict that is absent from the input window and determined by region.
    """

    def test_zero_concept_shift_is_byte_identical(self):
        """
        At concept_shift=0 the target stream must be the *same object* as the input
        obs (no copy, no flip) - the default world is unchanged.
        """
        world = OccupancyWorld(
            _polarity_two_region_cfg(n_channels=4, concept_shift=0.0),
            seed=0, n_steps=200,
        )
        lat = np.full(200, -45.0)  # region 1 (polarity -1): the group that would flip
        lon = np.zeros(200)
        obs = world.generate_obs_sequence(lat, lon, np.random.default_rng([0, 1]))
        region_ids = np.zeros(200, dtype=np.intp)
        region_ids[:] = region_at(-45.0, 0.0, world._region_map)
        target = world._apply_concept_shift(obs, region_ids)
        assert target is obs, "concept_shift=0 must return the obs object unchanged"

    def test_conflict_is_hidden_from_window_yet_region_resolved(self):
        """With concept_shift = 1 and regions of opposite polarity:
          (hidden)    the input windows of the two regions are statistically the same:
                      equal marginal occupancy and equal (positive) lag-1 autocorrelation;
          (resolved)  the persistence predictor, which is what a model that sees only
                      the window would learn, is accurate in the +1 region and worse
                      than chance in the -1 region.
        """
        T, C, delta = 6000, 8, 4
        world = OccupancyWorld(
            _polarity_two_region_cfg(n_channels=C, concept_shift=1.0),
            seed=0, n_steps=T,
        )
        lon = np.zeros(T)
        obs_pos = world.generate_obs_sequence(
            np.full(T, 45.0), lon, np.random.default_rng([0, 1]))   # region 0, +1
        obs_neg = world.generate_obs_sequence(
            np.full(T, -45.0), lon, np.random.default_rng([0, 2]))  # region 1, -1
        rid_pos = np.full(T, region_at(45.0, 0.0, world._region_map), dtype=np.intp)
        rid_neg = np.full(T, region_at(-45.0, 0.0, world._region_map), dtype=np.intp)
        tgt_pos = world._apply_concept_shift(obs_pos, rid_pos)  # unchanged (+1)
        tgt_neg = world._apply_concept_shift(obs_neg, rid_neg)  # flipped   (-1)

        # (hidden) input windows are statistically indistinguishable.
        assert abs(obs_pos.mean() - obs_neg.mean()) < 0.03, "window occupancy differs"
        ac_pos, ac_neg = _lag1_autocorr(obs_pos), _lag1_autocorr(obs_neg)
        assert ac_pos > 0.3 and ac_neg > 0.3, "both windows should be persistent"
        assert abs(ac_pos - ac_neg) < 0.05, "window autocorrelation differs by region"

        # (resolved) persistence is right in +1, wrong in -1 -> conflicting optima.
        acc_pos = _persistence_accuracy(obs_pos, tgt_pos, delta)
        acc_neg = _persistence_accuracy(obs_neg, tgt_neg, delta)
        assert acc_pos > 0.6, f"+1 region should be persistent, acc={acc_pos:.3f}"
        assert acc_neg < 0.4, f"−1 region target should be inverted, acc={acc_neg:.3f}"

    def test_zero_shift_regions_agree(self):
        """Null control: at concept_shift=0 both regions are persistent (no conflict)."""
        T, C, delta = 6000, 8, 4
        world = OccupancyWorld(
            _polarity_two_region_cfg(n_channels=C, concept_shift=0.0),
            seed=0, n_steps=T,
        )
        lon = np.zeros(T)
        obs_pos = world.generate_obs_sequence(
            np.full(T, 45.0), lon, np.random.default_rng([0, 1]))
        obs_neg = world.generate_obs_sequence(
            np.full(T, -45.0), lon, np.random.default_rng([0, 2]))
        rid_neg = np.full(T, region_at(-45.0, 0.0, world._region_map), dtype=np.intp)
        tgt_neg = world._apply_concept_shift(obs_neg, rid_neg)
        acc_pos = _persistence_accuracy(obs_pos, obs_pos, delta)
        acc_neg = _persistence_accuracy(obs_neg, tgt_neg, delta)
        assert acc_pos > 0.6 and acc_neg > 0.6, "both regions persistent at shift=0"
