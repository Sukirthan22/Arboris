"""
Arboris - artefact inventory and provenance (AGENTS.md §10).
Run:  python src/manifest.py

Every output file gets: path, bytes, sha256, generation timestamp, the git SHA of
the tree that produced it, and an explicit `synthetic` flag. If a number in the
deck cannot be traced back to a row in here, it does not get to exist.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("outputs")

TRACKED = [
    ("outputs/metrics.json",               "every number the UI displays"),
    ("outputs/diagnostics.json",           "hold-out diagnostics, uncertainty calibration, saturation"),
    ("outputs/cells_agb.geojson",          "200 m cells: predicted AGB, GEDI label, tree disagreement"),
    ("outputs/scatter.png",                "predicted vs observed, spatially-blocked hold-out"),
    ("outputs/feature_importance.png",     "random forest feature importance"),
    ("outputs/residuals.png",              "residuals vs predicted - shows high-biomass saturation"),
    ("outputs/uncertainty_calibration.png","tree disagreement vs actual hold-out error"),
    ("outputs/agb_map.png",                "static biomass map, fallback for the demo"),
    ("models/arboris_rf.pkl",              "fitted RandomForest + feature order + cell size"),
    ("data/arboris_training.csv",          "GEDI L4A labels joined to Sentinel-1/2 + SRTM features"),
    ("frontend/arboris_demo.html",         "self-contained demo page"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


sha = git("rev-parse", "--short", "HEAD")
dirty = bool(git("status", "--porcelain"))

entries, missing = [], []
for rel, desc in TRACKED:
    p = Path(rel)
    if not p.exists():
        missing.append(rel)
        continue
    entries.append({
        "path": rel,
        "description": desc,
        "bytes": p.stat().st_size,
        "sha256": sha256(p),
        "modified_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                                .isoformat(timespec="seconds"),
    })

metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))

manifest = {
    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_sha": sha,
    "git_tree_dirty": dirty,
    "synthetic": metrics.get("synthetic", None),
    "headline": {
        "total_co2e_mg": metrics["carbon"]["total_co2e_mg"],
        "mapped_area_ha": metrics["aoi"]["mapped_area_ha"],
        "r2_mean": metrics["r2_mean"],
        "rmse_mean_mg_ha": metrics["rmse_mean"],
    },
    "n_artefacts": len(entries),
    "artefacts": entries,
    "missing": missing,
}

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(f"git {sha}{' (DIRTY TREE)' if dirty else ''}  synthetic={manifest['synthetic']}")
for e in entries:
    print(f"  {e['sha256'][:12]}  {e['bytes']:>10,}  {e['path']}")
if missing:
    print("\nMISSING (not written to manifest):")
    for m in missing:
        print(f"  ! {m}")
print(f"\nwrote {OUT/'manifest.json'} - {len(entries)} artefacts, {len(missing)} missing")
