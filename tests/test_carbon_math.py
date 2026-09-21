"""
AGENTS.md §9 - the carbon arithmetic is the whole deliverable, so it is pinned.
These assertions are exact and never change.

Run:  pytest tests/ -q
No Earth Engine credentials required.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carbon.math import (  # noqa: E402
    AGB_TO_CO2E, CARBON_FRACTION, CO2_PER_C, CarbonSanityError,
    agb_to_carbon, carbon_to_co2e, cells_to_total_co2e, sanity_check,
)

CELL_AREA_HA = (200 * 200) / 10_000  # 4.0 ha, derived not typed


# ---------------------------------------------------------------- constants
def test_constants_are_the_documented_ones():
    assert CARBON_FRACTION == 0.47                     # IPCC default
    assert CO2_PER_C == pytest.approx(44.0 / 12.0)     # molecular weight ratio
    assert AGB_TO_CO2E == pytest.approx(1.7233, abs=1e-4)


# ---------------------------------------------------------------- §6.9 fixed values
def test_150_mg_ha_gives_70_5_carbon_and_258_5_co2e():
    c = float(agb_to_carbon(150.0))
    assert c == pytest.approx(70.5, abs=0.1)
    assert float(carbon_to_co2e(c)) == pytest.approx(258.5, abs=0.1)


def test_one_km2_at_100_mg_ha_totals_10_000_mg_not_1_000_000():
    """The hectares-vs-km² error, caught by arithmetic rather than by a judge."""
    s = cells_to_total_co2e(np.full(25, 100.0), CELL_AREA_HA)   # 25 × 4 ha = 1 km²
    assert s.area_ha == pytest.approx(100.0)
    assert s.total_agb_mg == pytest.approx(10_000.0)
    assert s.total_agb_mg != pytest.approx(1_000_000.0)


# ---------------------------------------------------------------- guards
def test_negative_biomass_raises():
    with pytest.raises(ValueError):
        agb_to_carbon(-1.0)
    with pytest.raises(ValueError):
        carbon_to_co2e(-1.0)


def test_nan_propagates_and_is_never_silently_zero():
    assert np.isnan(float(agb_to_carbon(np.nan)))
    assert np.isnan(float(carbon_to_co2e(np.nan)))


def test_nan_cells_are_excluded_from_totals_not_counted_as_zero():
    s = cells_to_total_co2e(np.array([100.0, np.nan, 100.0]), CELL_AREA_HA)
    assert s.n_cells == 2
    assert s.mean_agbd_mg_ha == pytest.approx(100.0)
    assert s.area_ha == pytest.approx(8.0)


def test_all_nan_raises_rather_than_reporting_zero():
    with pytest.raises(ValueError):
        cells_to_total_co2e(np.array([np.nan, np.nan]), CELL_AREA_HA)


def test_zero_or_negative_cell_area_raises():
    with pytest.raises(ValueError):
        cells_to_total_co2e(np.array([100.0]), 0.0)


# ---------------------------------------------------------------- sanity gate
def test_sanity_check_passes_on_correct_ratio():
    sanity_check(150.0, 258.5)          # must not raise


@pytest.mark.parametrize("wrong", [258.5 / 10_000, 258.5 * 100, 258.5 * 1.05])
def test_sanity_check_catches_unit_bugs(wrong):
    """m² vs ha, km² vs ha, and a 5 % drift all have to trip the guard."""
    with pytest.raises(CarbonSanityError):
        sanity_check(150.0, wrong)


def test_sanity_check_is_called_inside_the_total():
    """cells_to_total_co2e must not be able to return an unsanitised number."""
    s = cells_to_total_co2e(np.full(10, 132.0), CELL_AREA_HA)
    assert s.mean_co2e_mg_ha / s.mean_agbd_mg_ha == pytest.approx(AGB_TO_CO2E, rel=1e-9)


# ---------------------------------------------------------------- units end to end
def test_totals_are_internally_consistent():
    s = cells_to_total_co2e(np.array([50.0, 150.0, 250.0]), CELL_AREA_HA)
    assert s.total_agb_mg == pytest.approx(450.0 * CELL_AREA_HA)
    assert s.total_c_mg == pytest.approx(s.total_agb_mg * CARBON_FRACTION)
    assert s.total_co2e_mg == pytest.approx(s.total_c_mg * CO2_PER_C)
    assert s.mean_co2e_mg_ha == pytest.approx(s.total_co2e_mg / s.area_ha)
