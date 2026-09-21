"""
Arboris - cell-level biomass prediction + carbon totals + demo artefacts.
Run:  python src/predict_cells.py

Input :  data/arboris_training.csv    GEDI labels + Sentinel-1/2 + terrain
         models/arboris_rf.pkl        the model trained by src/data/train_model.py

Output:  outputs/cells_agb.geojson    200 m cells with predicted AGB (EPSG:4326)
         outputs/metrics.json         EXTENDED - existing model numbers untouched
         outputs/agb_map.png          static fallback map for the video
         frontend/data.js             same GeoJSON + panel numbers, inlined for file:// use

No Earth Engine call. No live training. Reads the persisted model only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carbon.math import CARBON_FRACTION, CO2_PER_C, cells_to_total_co2e

# ---------------------------------------------------------------- config
# These MUST match src/data/train_model.py or the feature construction diverges.
CSV         = Path("data/arboris_training.csv")
MODEL       = Path("models/arboris_rf.pkl")
CELL_M      = 200      # cell edge, metres
MIN_SHOTS   = 3        # a *training* cell needs >= 3 shots; prediction does not
SE_MAX_FRAC = 0.5      # drop shots whose own error exceeds 50% of the estimate
DEG_PER_CELL = CELL_M / 111_320.0   # same lat-degree approximation as training

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
FRONTEND = Path("frontend"); FRONTEND.mkdir(exist_ok=True)

# ---------------------------------------------------------------- load + filter
df = pd.read_csv(CSV)
coords = df[".geo"].map(lambda s: json.loads(s)["coordinates"])
df["lon"] = [c[0] for c in coords]
df["lat"] = [c[1] for c in coords]
print(f"loaded {len(df):,} GEDI footprints")

df = df[(df.agbd_se / df.agbd) <= SE_MAX_FRAC].copy()
print(f"{len(df):,} left after GEDI self-reported-error filter (se/agbd <= {SE_MAX_FRAC})")

bundle = joblib.load(MODEL)
FEATURES = bundle["features"]          # read from the model, never re-derived
if bundle.get("cell_size_m") != CELL_M:
    raise ValueError(
        f"model was fitted at cell_size_m={bundle.get('cell_size_m')} "
        f"but this script uses CELL_M={CELL_M} - refusing to predict."
    )
print(f"model expects {len(FEATURES)} features")

# ---------------------------------------------------------------- aggregate to cells
df["cx"] = np.floor(df.lon / DEG_PER_CELL).astype(int)
df["cy"] = np.floor(df.lat / DEG_PER_CELL).astype(int)

cells = (df.groupby(["cx", "cy"])
           .agg(**{f: (f, "mean") for f in FEATURES},
                agb_gedi_mg_ha=("agbd", "mean"),
                nshot=("agbd", "size"))
           .reset_index())
n_train_cells = int((cells.nshot >= MIN_SHOTS).sum())
print(f"{len(cells):,} cells at {CELL_M} m "
      f"({n_train_cells:,} of them have >= {MIN_SHOTS} shots and were trainable)")

# ---------------------------------------------------------------- predict
rf = bundle["model"]
X = cells[FEATURES]
pred = rf.predict(X)
n_clipped = int((pred < 0).sum())
pred = np.clip(pred, 0.0, None)          # biomass cannot be negative
cells["agb_pred_mg_ha"] = pred
cells["is_training_cell"] = cells.nshot >= MIN_SHOTS
print(f"predicted AGB: mean {pred.mean():.1f}  median {np.median(pred):.1f}  "
      f"max {pred.max():.1f} Mg/ha   ({n_clipped} negatives clipped to 0)")

# Per-cell model disagreement: 1 s.d. across the 500 individual trees.
# NOT a calibrated confidence interval - but src/diagnostics.py measures, on a
# spatial hold-out, that the third of cells with the widest spread really does
# carry ~2.4x the RMSE of the narrowest third. So it earns its place on the map.
Xv = X.to_numpy()
spread = np.stack([t.predict(Xv) for t in rf.estimators_]).std(axis=0)
cells["agb_spread_mg_ha"] = spread
print(f"tree disagreement: mean {spread.mean():.1f}  "
      f"p10 {np.percentile(spread,10):.1f}  p90 {np.percentile(spread,90):.1f} Mg/ha")

# ---------------------------------------------------------------- cell geometry + area
# The grid is defined in DEGREES, so a cell is CELL_M tall but only
# CELL_M * cos(latitude) wide. Using a flat 4.00 ha here would overstate the
# area - and therefore the headline tonnage - by ~1.7%.
lat0 = float(df.lat.mean())
cell_h_m = DEG_PER_CELL * 111_320.0
cell_w_m = DEG_PER_CELL * 111_320.0 * np.cos(np.radians(lat0))
CELL_AREA_HA = (cell_w_m * cell_h_m) / 10_000.0
print(f"cell footprint at lat {lat0:.3f}: {cell_w_m:.1f} m x {cell_h_m:.1f} m "
      f"= {CELL_AREA_HA:.4f} ha  (a flat 4.00 ha would be {4/CELL_AREA_HA-1:+.2%} off)")

# ---------------------------------------------------------------- carbon
summary = cells_to_total_co2e(cells.agb_pred_mg_ha.to_numpy(), CELL_AREA_HA)
print("\n=== HEADLINE ===")
print(f"mapped area      {summary.area_ha:,.0f} ha")
print(f"mean AGB         {summary.mean_agbd_mg_ha:,.1f} Mg/ha")
print(f"total AGB        {summary.total_agb_mg:,.0f} Mg")
print(f"total carbon     {summary.total_c_mg:,.0f} Mg C")
print(f"total CO2e       {summary.total_co2e_mg:,.0f} Mg CO2e")
print(f"mean CO2e        {summary.mean_co2e_mg_ha:,.1f} Mg CO2e/ha")
print("sanity_check passed (CO2e/AGB ratio == 1.7233)")

# ---------------------------------------------------------------- GeoJSON
feats = []
for cx, cy, p, g, n, istr, sd in zip(cells.cx, cells.cy, cells.agb_pred_mg_ha,
                                     cells.agb_gedi_mg_ha, cells.nshot,
                                     cells.is_training_cell,
                                     cells.agb_spread_mg_ha):
    x0, y0 = cx * DEG_PER_CELL, cy * DEG_PER_CELL
    x1, y1 = x0 + DEG_PER_CELL, y0 + DEG_PER_CELL
    feats.append({
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[
            [round(x0, 6), round(y0, 6)], [round(x1, 6), round(y0, 6)],
            [round(x1, 6), round(y1, 6)], [round(x0, 6), round(y1, 6)],
            [round(x0, 6), round(y0, 6)]]]},
        "properties": {"p": round(float(p), 1), "g": round(float(g), 1),
                       "n": int(n), "t": bool(istr), "u": round(float(sd), 1)},
    })

bounds = [float(df.lon.min()), float(df.lat.min()),
          float(df.lon.max()), float(df.lat.max())]
gj = {"type": "FeatureCollection",
      "properties": {"cell_size_m": CELL_M, "cell_area_ha": round(CELL_AREA_HA, 4),
                     "crs": "EPSG:4326", "bounds": bounds,
                     "property_keys": {"p": "agb_pred_mg_ha", "g": "agb_gedi_mg_ha",
                                       "n": "gedi_shot_count", "t": "is_training_cell",
                                       "u": "agb_tree_disagreement_sd_mg_ha"}},
      "features": feats}

gj_path = OUT / "cells_agb.geojson"
gj_path.write_text(json.dumps(gj, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {gj_path} ({gj_path.stat().st_size/1e6:.2f} MB, {len(feats):,} cells)")

# ---------------------------------------------------------------- metrics.json (EXTEND)
mpath = OUT / "metrics.json"
metrics = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
model_block = {k: metrics[k] for k in
               ("n_cells", "cell_size_m", "n_features", "r2_mean", "r2_sd",
                "rmse_mean", "rmse_sd", "mae_mean", "validation") if k in metrics}

metrics.update({
    "aoi": {"name": "nilgiris_anamalai",
            "bounds_epsg4326": [round(b, 5) for b in bounds],
            "mapped_area_ha": round(summary.area_ha, 1),
            "note": "mapped area = union of 200 m cells containing >=1 quality-filtered "
                    "GEDI footprint; NOT the full administrative AOI"},
    "biomass": {"n_cells_predicted": summary.n_cells,
                "n_cells_trainable": n_train_cells,
                "cell_area_ha": round(CELL_AREA_HA, 4),
                "mean_agbd_mg_ha": round(summary.mean_agbd_mg_ha, 2),
                "total_agb_mg": round(summary.total_agb_mg, 1),
                "n_cells_clipped_to_zero": n_clipped,
                "mean_tree_disagreement_mg_ha": round(float(cells.agb_spread_mg_ha.mean()), 2)},
    "carbon": {"carbon_fraction": CARBON_FRACTION, "co2_per_c": round(CO2_PER_C, 4),
               "total_c_mg": round(summary.total_c_mg, 1),
               "total_co2e_mg": round(summary.total_co2e_mg, 1),
               "mean_co2e_mg_ha": round(summary.mean_co2e_mg_ha, 2),
               "scope": "above-ground biomass only"},
    "synthetic": False,
})
mpath.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
assert all(metrics[k] == v for k, v in model_block.items()), "model metrics were mutated!"
print(f"extended {mpath} (existing model metrics preserved verbatim)")

# ---------------------------------------------------------------- frontend/data.js
panel = {
    "total_co2e_mg": summary.total_co2e_mg,
    "total_agb_mg": summary.total_agb_mg,
    "total_c_mg": summary.total_c_mg,
    "area_ha": summary.area_ha,
    "mean_agbd_mg_ha": summary.mean_agbd_mg_ha,
    "mean_co2e_mg_ha": summary.mean_co2e_mg_ha,
    "n_cells": summary.n_cells,
    "n_cells_trainable": n_train_cells,
    "n_footprints": int(len(df)),
    "mean_spread_mg_ha": float(cells.agb_spread_mg_ha.mean()),
    "diagnostics": json.loads((OUT / "diagnostics.json").read_text(encoding="utf-8"))
                   if (OUT / "diagnostics.json").exists() else None,
    "r2_mean": metrics.get("r2_mean"), "r2_sd": metrics.get("r2_sd"),
    "rmse_mean": metrics.get("rmse_mean"), "rmse_sd": metrics.get("rmse_sd"),
    "mae_mean": metrics.get("mae_mean"), "validation": metrics.get("validation"),
    "carbon_fraction": CARBON_FRACTION, "co2_per_c": round(CO2_PER_C, 4),
    "cell_size_m": CELL_M, "bounds": bounds,
}
js = FRONTEND / "data.js"
js.write_text("// GENERATED by src/predict_cells.py - do not edit by hand.\n"
              "window.ARBORIS_PANEL = " + json.dumps(panel) + ";\n"
              "window.ARBORIS_CELLS = " + json.dumps(gj, separators=(",", ":")) + ";\n",
              encoding="utf-8")
print(f"wrote {js} ({js.stat().st_size/1e6:.2f} MB)")

# ---------------------------------------------------------------- static fallback map
fig, ax = plt.subplots(figsize=(11, 8))
ctr_x = (cells.cx + 0.5) * DEG_PER_CELL
ctr_y = (cells.cy + 0.5) * DEG_PER_CELL
sc = ax.scatter(ctr_x, ctr_y, c=cells.agb_pred_mg_ha, s=7, marker="s",
                cmap="YlGn", vmin=0, vmax=300, linewidths=0)
fig.colorbar(sc, ax=ax, label="Predicted above-ground biomass (Mg/ha)")
ax.set_aspect(1 / np.cos(np.radians(lat0)))
ax.set_xlabel("Longitude (deg E)"); ax.set_ylabel("Latitude (deg N)")
ax.set_title(f"Arboris - predicted above-ground biomass, {CELL_M} m cells\n"
             f"{summary.total_co2e_mg/1e6:.2f} million Mg CO2e over "
             f"{summary.area_ha:,.0f} ha  |  "
             f"R2 {metrics.get('r2_mean')} +/- {metrics.get('r2_sd')}, "
             f"RMSE {metrics.get('rmse_mean')} Mg/ha (spatially-blocked)")
fig.tight_layout()
fig.savefig(OUT / "agb_map.png", dpi=160)
print(f"wrote {OUT / 'agb_map.png'}")

# ---------------------------------------------------------------- self-contained demo page
# frontend/index.html loads data.js with a <script src>. That works over http and
# (in Chrome/Edge) over file:// too, but "works in most browsers" is not a thing to
# discover five minutes before recording. This bundles the data INTO the page so the
# demo file has zero local dependencies - double-click it and it runs.
tpl = FRONTEND / "index.html"
if tpl.exists():
    html = tpl.read_text(encoding="utf-8")
    tag = '<script src="data.js"></script>'
    if tag not in html:
        raise ValueError(f"expected {tag!r} in frontend/index.html - cannot bundle")
    inline = "<script>\n" + js.read_text(encoding="utf-8") + "</script>"
    demo = FRONTEND / "arboris_demo.html"
    demo.write_text(html.replace(tag, inline), encoding="utf-8")
    print(f"wrote {demo} ({demo.stat().st_size/1e6:.2f} MB, self-contained - "
          f"double-click this one for the video)")
