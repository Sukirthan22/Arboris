"""
Phase 3 — crown scoring, config pinning, and the /api/crowns endpoint.
No detectree2 / detectron2 / GPU required: inference outputs are precomputed.

Run:  pytest tests/ -q
"""
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.crowns import (  # noqa: E402
    choose_threshold, load_crowns_config, pooled, score, tile_paths, xml_truth,
)

CRS = "EPSG:32611"


def gdf(boxes, scores=None):
    d = {"geometry": [box(*b) for b in boxes]}
    if scores is not None:
        d["Confidence_score"] = scores
    return gpd.GeoDataFrame(d, crs=CRS)


# ── config: the notebook's exact settings, pinned ─────────────────────────────

def test_config_pins_notebook_settings():
    c = load_crowns_config()
    assert c["model_version"] == "250312_flexi"
    assert c["tiling"] == dict(tile_width=41, tile_height=41, buffer=0,
                               full_coverage=True, dtype_bool=True)
    assert c["score_thresh_test"] == 0.05
    assert c["confidence_threshold"] == 0.1
    assert c["match_iou"] == 0.4
    assert c["tune_tiles"] == [1, 2, 3, 4, 5, 6, 7]
    assert c["test_tiles"] == [8, 9, 10]
    assert not set(c["tune_tiles"]) & set(c["test_tiles"]), "held-out tiles leaked into tuning"
    assert c["confidence_threshold"] in c["threshold_candidates"]


# ── score ──────────────────────────────────────────────────────────────────────

def test_score_exact_match_and_threshold_filter():
    truth = gdf([(0, 0, 10, 10), (20, 0, 30, 10)])
    pred = gdf([(0, 0, 10, 10), (20, 0, 30, 10)], scores=[0.9, 0.05])
    assert score(pred, truth, 0.1, 0.4) == (1, 1, 2)   # low-score crown dropped
    assert score(pred, truth, 0.0, 0.4) == (2, 2, 2)


def test_score_iou_boundary():
    truth = gdf([(0, 0, 10, 10)])
    at = gdf([(0, 0, 10, 4)], scores=[1.0])              # IoU = 40/100 = 0.40 exactly
    below = gdf([(0, 0, 10, 3.9)], scores=[1.0])         # IoU = 0.39
    assert score(at, truth, 0.1, 0.4)[0] == 1
    assert score(below, truth, 0.1, 0.4)[0] == 0


def test_score_is_one_to_one():
    truth = gdf([(0, 0, 10, 10)])
    pred = gdf([(0, 0, 10, 10), (0, 0, 10, 9)], scores=[0.9, 0.8])
    assert score(pred, truth, 0.1, 0.4) == (1, 2, 1)     # second box is a false positive


def test_score_uses_envelope_of_predicted_polygon():
    from shapely.geometry import Polygon
    truth = gdf([(0, 0, 10, 10)])
    diamond = Polygon([(5, 0), (10, 5), (5, 10), (0, 5)])  # IoU 0.5 as polygon, 1.0 as envelope
    pred = gpd.GeoDataFrame({"Confidence_score": [1.0]}, geometry=[diamond], crs=CRS)
    assert score(pred, truth, 0.1, 0.9)[0] == 1


# ── pooled + threshold choice ─────────────────────────────────────────────────

def test_pooled_micro_averages_counts():
    truths = {1: gdf([(0, 0, 10, 10)]), 2: gdf([(0, 0, 10, 10), (20, 0, 30, 10), (40, 0, 50, 10)])}
    preds = {1: gdf([(0, 0, 10, 10), (60, 0, 70, 10)], [0.9, 0.9]),
             2: gdf([(0, 0, 10, 10)], [0.9])}
    r = pooled(preds, truths, 0.1, 0.4, [1, 2])
    assert (r["matched"], r["n_pred"], r["n_true"]) == (2, 3, 4)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 4)
    assert r["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))


def test_choose_threshold_ignores_test_tiles():
    tree = (0, 0, 10, 10)
    # tune tile: the true crown scores 0.15, so only thr=0.1 finds it
    # test tile: a false positive at 0.15 would push the choice to 0.2 if it were consulted
    preds = {1: gdf([tree], [0.15]), 8: gdf([tree, (50, 50, 60, 60)], [0.9, 0.15])}
    truths = {1: gdf([tree]), 8: gdf([tree])}
    cfg = dict(threshold_candidates=[0.1, 0.2], match_iou=0.4, tune_tiles=[1], test_tiles=[8])
    assert choose_threshold(preds, truths, cfg) == 0.1


# ── xml_truth on the real demo tile ───────────────────────────────────────────

def test_xml_truth_demo8():
    img, xml = tile_paths(load_crowns_config(), 8)
    if not img.exists():
        pytest.skip("data/raw/demo8.tif not present")
    import rasterio
    t = xml_truth(img, xml)
    with rasterio.open(img) as s:
        bounds, crs = box(*s.bounds), s.crs
    assert len(t) == 8                                   # notebook cell 19: "vs 8 true"
    assert t.crs == crs
    assert all(g.area > 0 and bounds.buffer(1e-6).contains(g) for g in t.geometry)


# ── API ───────────────────────────────────────────────────────────────────────

def test_api_crowns():
    from fastapi.testclient import TestClient
    from api.main import app
    r = TestClient(app).get("/api/crowns")
    assert r.status_code == 200
    body = r.json()
    fc = body["geojson"]
    assert body["n_crowns"] == fc["properties"]["n_crowns"] == len(fc["features"])
    assert body["n_crowns"] == len(gpd.read_file(ROOT / load_crowns_config()["demo_output"]))
    assert fc["properties"]["confidence_threshold"] == 0.1
    lon0, lat0, lon1, lat1 = fc["properties"]["crowns_bounds"]
    assert -180 <= lon0 < lon1 <= 180 and -90 <= lat0 < lat1 <= 90   # reprojected to EPSG:4326
    f = fc["features"][0]["properties"]
    assert {"crown_id", "Confidence_score", "area_m2"} <= f.keys()
