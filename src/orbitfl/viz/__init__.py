"""Constellation figures.

This package only draws. It reads the config and calls the simulator's own
geometry code (orbitfl.geometry), so the figures show the same satellites,
links and radii that the experiments use.
"""

from __future__ import annotations

__all__ = ["constellation_data", "constellation_figure", "check_constellation", "earth", "palettes"]
