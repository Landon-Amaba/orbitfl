"""Colour definitions for the constellation figures.

Palettes are defined once here so that every figure variant colours a given
orbital plane the same way.
"""

from __future__ import annotations

# Paul Tol's "discrete rainbow", 12 colours (SRON/EPS/TN/09-002, "Colour
# schemes", fig. 13). Colour-blind safe; neighbouring plane indices also differ
# in greyscale.
TOL_RAINBOW_12: tuple[str, ...] = (
    "#D1BBD7",  # 0
    "#AE76A3",  # 1
    "#882E72",  # 2
    "#1965B0",  # 3
    "#5289C7",  # 4
    "#7BAFDE",  # 5
    "#4EB265",  # 6
    "#90C987",  # 7
    "#CAE0AB",  # 8
    "#F7F056",  # 9
    "#F6C141",  # 10
    "#F1932D",  # 11
)

# MEO tier: Tol "bright" scheme (same technical note, fig. 1), distinct from the
# LEO rainbow.
TOL_BRIGHT_4: tuple[str, ...] = (
    "#EE6677",  # MEO 0 - red
    "#228833",  # MEO 1 - green
    "#4477AA",  # MEO 2 - blue
    "#CCBB44",  # MEO 3 - yellow
)

# LEOs with assignment == -1 (no MEO in line of sight): neutral grey.
UNASSIGNED_COLOR: str = "#BBBBBB"

# Non-data ink: marker edges, ISL edges, graticule, footprint rings.
MARKER_EDGE: str = "#1A1A1A"
ISL_EDGE: str = "#FFFFFF"
GRATICULE: str = "#FFFFFF"
FOOTPRINT_RING: str = "#FFFFFF"


def plane_colors(n_planes: int) -> list[str]:
    """One colour per LEO orbital plane.

    Parameters
    ----------
    n_planes : int  number of orbital planes (12 in the thesis constellation)

    Returns
    -------
    list[str], length n_planes  hex colour strings

    Raises
    ------
    ValueError
        If n_planes exceeds the 12 colours available. Reusing colours would give
        two different planes the same colour.
    """
    if n_planes > len(TOL_RAINBOW_12):
        raise ValueError(
            f"n_planes={n_planes} exceeds the {len(TOL_RAINBOW_12)} colour-blind-safe "
            "colours in TOL_RAINBOW_12; add a cited scheme rather than cycling colours."
        )
    return list(TOL_RAINBOW_12[:n_planes])


def meo_colors(n_meo: int) -> list[str]:
    """One colour per MEO.

    Parameters
    ----------
    n_meo : int  number of MEOs (4 in the thesis constellation)

    Returns
    -------
    list[str], length n_meo  hex colour strings

    Raises
    ------
    ValueError  if n_meo exceeds the 4 available colours.
    """
    if n_meo > len(TOL_BRIGHT_4):
        raise ValueError(
            f"n_meo={n_meo} exceeds the {len(TOL_BRIGHT_4)} colours in TOL_BRIGHT_4."
        )
    return list(TOL_BRIGHT_4[:n_meo])
