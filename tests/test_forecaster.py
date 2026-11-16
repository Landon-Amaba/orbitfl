"""Tests for src/orbitfl/models/forecaster.py.

F1  output shape: (B, C) logits from (B, W, C) observations and (B, 4) positions.
F2  use_position=False: the output does not depend on the position input.
F3  the model can overfit 50 samples: BCE loss < 0.05 after 300 Adam steps.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from orbitfl.models.forecaster import OccupancyForecaster, positional_encoding

# --- Helpers ---

def _make_batch(
    B: int = 4,
    W: int = 16,
    C: int = 8,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (obs, pos) tensors with the standard shapes."""
    rng = np.random.default_rng(seed)
    obs = torch.from_numpy(rng.integers(0, 2, (B, W, C)).astype(np.float32))
    pos = torch.from_numpy(rng.standard_normal((B, 4)).astype(np.float32))
    return obs, pos


# --- F1 - correct output shapes ---

class TestOutputShapes:
    """forward() must always return (B, C) logits."""

    @pytest.mark.parametrize("B,W,C,hidden", [
        (1, 8, 4, 16),
        (4, 16, 32, 64),
        (16, 32, 8, 32),
    ])
    def test_shape_with_position(self, B: int, W: int, C: int, hidden: int):
        model = OccupancyForecaster(n_channels=C, window=W, use_position=True, hidden=hidden)
        obs, pos = _make_batch(B, W, C)
        logits = model(obs, pos)
        assert logits.shape == (B, C), f"Expected ({B},{C}), got {logits.shape}"

    @pytest.mark.parametrize("B,W,C", [(2, 16, 32), (8, 8, 16)])
    def test_shape_without_position(self, B: int, W: int, C: int):
        model = OccupancyForecaster(n_channels=C, window=W, use_position=False)
        obs, _ = _make_batch(B, W, C)
        logits = model(obs)
        assert logits.shape == (B, C), f"Expected ({B},{C}), got {logits.shape}"

    def test_output_is_logits_not_probs(self):
        """Output is unbounded (not sigmoid-squashed); values outside [0, 1] expected."""
        model = OccupancyForecaster(n_channels=16, window=16, use_position=True)
        obs, pos = _make_batch(B=32, W=16, C=16)
        logits = model(obs, pos)
        # After random init, some logits are almost certainly outside [0, 1]
        assert logits.detach().abs().max().item() > 0.0, "Logits should be non-trivial"

    def test_dtype_preserved(self):
        """Output dtype matches input dtype (float32 in -> float32 out)."""
        model = OccupancyForecaster(n_channels=8, window=8, use_position=True)
        obs, pos = _make_batch(B=4, W=8, C=8)
        assert model(obs, pos).dtype == torch.float32


# --- F2 - use_position=False ignores pos input ---

class TestPositionIgnored:
    """When use_position=False, the pos tensor must have zero effect on output."""

    def test_different_pos_same_output(self):
        model = OccupancyForecaster(n_channels=8, window=16, use_position=False)
        model.eval()
        obs, pos_a = _make_batch(B=4, W=16, C=8, seed=0)
        _, pos_b = _make_batch(B=4, W=16, C=8, seed=999)
        # Guarantee pos_a != pos_b
        assert not torch.allclose(pos_a, pos_b)
        with torch.no_grad():
            out_a = model(obs, pos_a)
            out_b = model(obs, pos_b)
        torch.testing.assert_close(out_a, out_b, msg="use_position=False must ignore pos")

    def test_pos_none_matches_pos_provided(self):
        """Passing pos=None vs passing a pos tensor should give the same output."""
        model = OccupancyForecaster(n_channels=8, window=16, use_position=False)
        model.eval()
        obs, pos = _make_batch(B=4, W=16, C=8)
        with torch.no_grad():
            out_with = model(obs, pos)
            out_none = model(obs, None)
        torch.testing.assert_close(out_with, out_none)


# --- F3 - overfit 50 samples to ~0 loss ---

class TestOverfit:
    """
    A model with enough capacity must be able to overfit 50 synthetic samples
    to near-zero BCE loss in 300 Adam steps.  This is a learnability smoke test,
    not a generalization test.
    """

    def _run_overfit(self, use_position: bool) -> float:
        """Train on 50 fixed samples; return final BCE loss."""
        B, W, C = 50, 16, 8
        rng = np.random.default_rng(42)
        obs = torch.from_numpy(rng.integers(0, 2, (B, W, C)).astype(np.float32))
        pos = torch.from_numpy(rng.standard_normal((B, 4)).astype(np.float32))
        target = torch.from_numpy(rng.integers(0, 2, (B, C)).astype(np.float32))

        model = OccupancyForecaster(
            n_channels=C, window=W, use_position=use_position, hidden=64
        )
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(300):
            opt.zero_grad()
            logits = model(obs, pos if use_position else None)
            loss = loss_fn(logits, target)
            loss.backward()
            opt.step()

        return float(loss.item())

    def test_overfit_with_position(self):
        """use_position=True: final BCE loss < 0.05 on 50 samples."""
        final_loss = self._run_overfit(use_position=True)
        assert final_loss < 0.05, (
            f"Model with position did not overfit: final BCE = {final_loss:.4f} ≥ 0.05"
        )

    def test_overfit_without_position(self):
        """use_position=False: same capacity -> should still overfit 50 samples."""
        final_loss = self._run_overfit(use_position=False)
        assert final_loss < 0.05, (
            f"Model without position did not overfit: final BCE = {final_loss:.4f} ≥ 0.05"
        )


# --- Positional encoding ---

class TestPositionalEncoding:
    def test_shape_and_dtype(self):
        pe = positional_encoding(45.0, -90.0)
        assert pe.shape == (4,)
        assert pe.dtype == np.float32

    def test_poles_finite(self):
        """positional_encoding must return finite values at the poles."""
        assert np.all(np.isfinite(positional_encoding(90.0, 0.0)))
        assert np.all(np.isfinite(positional_encoding(-90.0, 0.0)))

    def test_dateline_continuous(self):
        """+/-180 deg longitude should map to identical sin/cos lon values."""
        pe_west = positional_encoding(0.0, -180.0)
        pe_east = positional_encoding(0.0, 180.0)
        np.testing.assert_allclose(pe_west[2:], pe_east[2:], atol=1e-6)

    def test_values_in_range(self):
        """All components must be in [-1, 1]."""
        for lat, lon in [(0, 0), (45, 90), (-53, -120), (90, 0), (-90, 180)]:
            pe = positional_encoding(float(lat), float(lon))
            assert np.all(np.abs(pe) <= 1.0 + 1e-7), f"Out of range at ({lat}, {lon}): {pe}"
