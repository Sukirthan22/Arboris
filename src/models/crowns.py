"""
Arboris — Phase 3: tree-crown detection on the demo tile.

Ported from notebooks/detectree2.ipynb. Every tunable lives under `crowns:` in
config/config.yaml; nothing numeric is hardcoded here.

What was actually shipped, and why:
  * detectree2 pretrained weights (250312_flexi) used as-is. NO fine-tune: fine-tuning
    on 8 m training tiles sliced 10-13 m crowns apart and scored worse than the base
    weights on held-out tile 8 (see docs/validation_report.md §5).
  * Whole-image tiling: one 41 m tile covers each ~40 m NEON image, buffer 0.
  * Inference at a low score floor, then one confidence cutoff chosen on tiles 1-7 only.
  * Greedy 1-to-1 box matching at IoU >= match_iou, pooled over held-out tiles 8-10.

Scope: per-tree resolution on the demo tile only. Never across the AOI (AGENTS.md I7).

detectree2 / detectron2 need a CUDA source build and are NOT part of the local env.
They are imported lazily inside `predict_crowns`, so the scoring functions and the API
work without them. The demo output is precomputed on Colab.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping

import geopandas as gpd
import rasterio
import yaml
from shapely.geometry import box

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
SCORE_COL = "Confidence_score"  # column name detectree2 writes


def load_crowns_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Return the `crowns:` section of config.yaml. Raises KeyError if it is missing."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["crowns"]


def tile_paths(cfg: Mapping[str, Any], t: int) -> tuple[Path, Path]:
    """(image .tif, box-label .xml) for tile number `t`."""
    base = REPO_ROOT / cfg["data_dir"] / f"{cfg['tile_prefix']}{t}"
    return base.with_suffix(".tif"), base.with_suffix(".xml")


# ── inference (Colab / CUDA only) ──────────────────────────────────────────────

def predict_crowns(img: str | Path, weights: str | Path, out_dir: str | Path,
                   cfg: Mapping[str, Any]) -> gpd.GeoDataFrame:
    """Tile `img`, run detectree2 with `weights`, stitch and de-duplicate crowns.

    Returns crown polygons in the image CRS (metres) with a `Confidence_score` column
    (0-1). No confidence filtering beyond cfg['score_thresh_test'] happens here.
    """
    from detectron2.engine import DefaultPredictor
    from detectree2.models.outputs import clean_crowns, project_to_geojson, stitch_crowns
    from detectree2.models.predict import predict_on_data
    from detectree2.models.train import setup_cfg
    from detectree2.preprocessing.tiling import tile_data

    out = str(out_dir).rstrip("/\\") + "/"
    tiling = cfg["tiling"]
    tile_data(str(img), out, buffer=tiling["buffer"],
              tile_width=tiling["tile_width"], tile_height=tiling["tile_height"],
              full_coverage=tiling["full_coverage"], dtype_bool=tiling["dtype_bool"])
    d2cfg = setup_cfg(update_model=str(weights))
    d2cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg["score_thresh_test"]
    predict_on_data(out, out_folder=out + "predictions/", predictor=DefaultPredictor(d2cfg))
    project_to_geojson(out, out + "predictions/", out + "predictions_geo/")  # pixel → map coords
    crowns = stitch_crowns(out + "predictions_geo/", cfg["stitch_shift"])
    return clean_crowns(crowns, cfg["dedup_iou"], confidence=0)


# ── ground truth + scoring ─────────────────────────────────────────────────────

def xml_truth(img: str | Path, xml: str | Path) -> gpd.GeoDataFrame:
    """NEON Pascal-VOC box labels (pixel coords) → box polygons in the image CRS."""
    with rasterio.open(img) as s:
        T, crs = s.transform, s.crs
    g = []
    for o in ET.parse(xml).findall("object"):
        b = o.find("bndbox")
        x0, y0, x1, y1 = [float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        (a, d), (c, e) = T @ (x0, y0), T @ (x1, y1)
        g.append(box(a, e, c, d))
    return gpd.GeoDataFrame(geometry=g, crs=crs)


def score(pred: gpd.GeoDataFrame, truth: gpd.GeoDataFrame, thr: float,
          iou: float) -> tuple[int, int, int]:
    """Greedy 1-to-1 match of predicted crown envelopes to truth boxes.

    Keeps predictions with Confidence_score >= `thr`, compares their bounding boxes
    (truth is boxes only) and counts a match when the best unused IoU >= `iou`.
    Returns (matched, n_pred, n_true).
    """
    p = pred[pred[SCORE_COL] >= thr].reset_index(drop=True)
    p = p.set_geometry(p.envelope)
    used, m = set(), 0
    for g in p.geometry:
        best, bj = 0.0, None
        for j, t in truth.geometry.items():
            if j in used or not g.intersects(t):
                continue
            v = g.intersection(t).area / g.union(t).area
            if v > best:
                best, bj = v, j
        if best >= iou:
            m += 1
            used.add(bj)
    return m, len(p), len(truth)


def pooled(preds: Mapping[int, gpd.GeoDataFrame], truths: Mapping[int, gpd.GeoDataFrame],
           thr: float, iou: float, tiles: Iterable[int]) -> dict[str, float | int]:
    """Micro-averaged precision / recall / F1 over `tiles` (counts summed, then divided)."""
    M = P = T = 0
    for t in tiles:
        m, p, n = score(preds[t], truths[t], thr, iou)
        M += m
        P += p
        T += n
    pr, rc = M / max(P, 1), M / T
    f1 = 2 * pr * rc / max(pr + rc, 1e-9)
    return dict(thr=thr, iou=iou, matched=M, n_pred=P, n_true=T,
                precision=pr, recall=rc, f1=f1)


def choose_threshold(preds: Mapping[int, gpd.GeoDataFrame],
                     truths: Mapping[int, gpd.GeoDataFrame], cfg: Mapping[str, Any]) -> float:
    """Pick the confidence cutoff with the best pooled F1 on cfg['tune_tiles'] only."""
    return max(cfg["threshold_candidates"],
               key=lambda th: pooled(preds, truths, th, cfg["match_iou"], cfg["tune_tiles"])["f1"])


# ── the notebook, end to end (Colab / CUDA only) ───────────────────────────────

def run_phase3(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Predict tiles, choose the cutoff on tune tiles, score held-out tiles, write the demo."""
    cfg = cfg or load_crowns_config()
    weights = REPO_ROOT / cfg["weights_path"]
    preds, truths = {}, {}
    for t in sorted(set(cfg["tune_tiles"]) | set(cfg["test_tiles"])):
        img, xml = tile_paths(cfg, t)
        truths[t] = xml_truth(img, xml)
        preds[t] = predict_crowns(img, weights, REPO_ROOT / cfg["work_dir"] / f"pred_base{t}", cfg)

    best = choose_threshold(preds, truths, cfg)
    if best != cfg["confidence_threshold"]:
        raise ValueError(f"tuned threshold {best} != config confidence_threshold "
                         f"{cfg['confidence_threshold']}; update config deliberately")
    test = pooled(preds, truths, best, cfg["match_iou"], cfg["test_tiles"])
    log.info("Phase 3 held-out: %s", test)

    demo = preds[cfg["demo_tile"]]
    demo = demo[demo[SCORE_COL] >= best]
    out = REPO_ROOT / cfg["demo_output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    demo.to_file(out, driver="GeoJSON")
    log.info("%d predicted crowns vs %d true on tile %d → %s",
             len(demo), len(truths[cfg["demo_tile"]]), cfg["demo_tile"], out)
    return dict(threshold=best, test=test, n_demo_crowns=len(demo))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run_phase3())
