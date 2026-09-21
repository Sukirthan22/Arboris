"""
Arboris — Phase 4: AGB -> carbon -> CO2e, with unit guards.

Two multiplications. They are the whole deliverable, so they are made
unimpeachable rather than clever.

Sources for the constants:
  carbon fraction 0.47  — IPCC 2006 Guidelines, Vol.4 Ch.4, default carbon
                          fraction of oven-dry above-ground biomass
  44/12 = 3.6667        — molecular weight ratio CO2 : C

Scope: ABOVE-GROUND biomass only. No below-ground, no soil carbon, no dead wood.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Final

import numpy as np

CARBON_FRACTION: Final[float] = 0.47
CO2_PER_C: Final[float] = 44.0 / 12.0          # 3.66666...
AGB_TO_CO2E: Final[float] = CARBON_FRACTION * CO2_PER_C   # ~1.7233
SANITY_TOLERANCE: Final[float] = 0.01          # 1 %


class CarbonSanityError(ValueError):
    """Raised when the AGB -> CO2e ratio is not ~1.7233, i.e. a unit bug."""


@dataclass(frozen=True)
class CarbonSummary:
    """Every field carries its unit in the name. Mg = metric tonne."""
    n_cells: int
    area_ha: float
    mean_agbd_mg_ha: float
    total_agb_mg: float
    total_c_mg: float
    total_co2e_mg: float
    mean_co2e_mg_ha: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def agb_to_carbon(agb_mg_ha, fraction: float = CARBON_FRACTION):
    """Mg dry above-ground biomass / ha -> Mg carbon / ha.

    Raises ValueError on negative input. NaN propagates (it is NOT coerced to 0 —
    a missing pixel must stay missing all the way to the report).
    """
    arr = np.asarray(agb_mg_ha, dtype="float64")
    if np.any(arr < 0):
        raise ValueError(
            f"negative biomass passed to agb_to_carbon: min={np.nanmin(arr)}. "
            "Biomass cannot be negative — clip predictions to 0 before calling."
        )
    return arr * fraction


def carbon_to_co2e(c_mg_ha, ratio: float = CO2_PER_C):
    """Mg carbon / ha -> Mg CO2-equivalent / ha."""
    arr = np.asarray(c_mg_ha, dtype="float64")
    if np.any(arr < 0):
        raise ValueError(f"negative carbon passed to carbon_to_co2e: min={np.nanmin(arr)}")
    return arr * ratio


def sanity_check(mean_agbd_mg_ha: float, mean_co2e_mg_ha: float) -> None:
    """The two constants combine to ~1.7233, always.

    A forest at 150 Mg/ha must land near 258.5 Mg CO2e/ha. If it does not, the bug
    is units (hectares vs km2, or m2 vs hectares), not the model.
    """
    if mean_agbd_mg_ha <= 0:
        raise CarbonSanityError(
            f"mean AGB must be > 0 to sanity-check, got {mean_agbd_mg_ha}"
        )
    observed = mean_co2e_mg_ha / mean_agbd_mg_ha
    if abs(observed - AGB_TO_CO2E) / AGB_TO_CO2E > SANITY_TOLERANCE:
        raise CarbonSanityError(
            f"CO2e/AGB ratio is {observed:.4f}, expected {AGB_TO_CO2E:.4f} "
            f"(+/-{SANITY_TOLERANCE:.0%}). This is a UNIT bug, not a model bug — "
            "check pixel/cell area in hectares."
        )


def cells_to_total_co2e(agb_mg_ha: np.ndarray, cell_area_ha: float) -> CarbonSummary:
    """Per-cell AGB density (Mg/ha) -> totals over the mapped area.

    `cell_area_ha` must be derived from the cell edge length, never typed as a
    literal:  cell_area_ha = (edge_m * edge_m) / 10_000.
    That single line is the one that kills projects (failure mode #2).

    NaN cells are excluded from every total and from n_cells.
    """
    arr = np.asarray(agb_mg_ha, dtype="float64")
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        raise ValueError("no valid (non-NaN) cells passed to cells_to_total_co2e")
    if cell_area_ha <= 0:
        raise ValueError(f"cell_area_ha must be > 0, got {cell_area_ha}")

    n_cells = int(valid.size)
    area_ha = n_cells * cell_area_ha
    mean_agbd_mg_ha = float(valid.mean())

    total_agb_mg = float((valid * cell_area_ha).sum())
    total_c_mg = float(agb_to_carbon(total_agb_mg))
    total_co2e_mg = float(carbon_to_co2e(total_c_mg))
    mean_co2e_mg_ha = total_co2e_mg / area_ha

    sanity_check(mean_agbd_mg_ha, mean_co2e_mg_ha)

    return CarbonSummary(
        n_cells=n_cells,
        area_ha=area_ha,
        mean_agbd_mg_ha=mean_agbd_mg_ha,
        total_agb_mg=total_agb_mg,
        total_c_mg=total_c_mg,
        total_co2e_mg=total_co2e_mg,
        mean_co2e_mg_ha=mean_co2e_mg_ha,
    )
