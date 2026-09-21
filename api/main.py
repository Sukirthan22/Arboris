"""
Arboris API — static artefact serving. Reads only from outputs/ (AGENTS.md I6).
Never imports a model, never calls Earth Engine.

Run:  uvicorn api.main:app --reload
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parents[1]
with open(REPO_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
    CROWNS_CFG = yaml.safe_load(f)["crowns"]

app = FastAPI(title="Arboris")
app.add_middleware(CORSMiddleware, allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["GET"], allow_headers=["*"])


@app.get("/api/crowns")
def crowns() -> dict:
    """Predicted crowns on the demo tile (EPSG:4326) plus the tree count.

    Per-tree resolution on the demo tile only — not across the AOI.
    """
    path = REPO_ROOT / CROWNS_CFG["demo_output"]
    if not path.exists():
        raise HTTPException(503, f"crowns artefact missing: {CROWNS_CFG['demo_output']}")

    gdf = gpd.read_file(path)
    thr = CROWNS_CFG["confidence_threshold"]
    if len(gdf) and gdf["Confidence_score"].min() < thr:
        # The file must already be filtered at the configured cutoff; never re-filter silently.
        raise HTTPException(500, f"{path.name} contains crowns below confidence_threshold {thr}")

    gdf = gdf.reset_index(drop=True)
    gdf["crown_id"] = gdf.index.astype(int)
    gdf["area_m2"] = gdf.geometry.area.round(2)       # native UTM CRS, metres
    crowns_bounds = [float(v) for v in gdf.to_crs(4326).total_bounds] if len(gdf) else None

    fc = json.loads(gdf.to_crs(4326).to_json())
    fc["properties"] = {
        "tile_id": f"{CROWNS_CFG['tile_prefix']}{CROWNS_CFG['demo_tile']}",
        "crowns_bounds": crowns_bounds,          # extent of detected crowns, lon/lat
        "n_crowns": len(gdf),
        "model_version": CROWNS_CFG["model_version"],
        "confidence_threshold": thr,
        "benchmark": CROWNS_CFG["benchmark"],
        "scope": "per-tree resolution on the demo tile",
    }
    return {"n_crowns": len(gdf), "geojson": fc}
