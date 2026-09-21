# Arboris — Methodology

**AOI:** Nilgiris / Anamalai hills, Western Ghats, Tamil Nadu
(EPSG:4326 bounds `76.85031, 10.25034 → 77.24984, 10.54970`)
**Scope:** above-ground biomass only — no below-ground, soil, dead-wood or litter carbon.
**Every number below comes from `outputs/metrics.json` or `outputs/diagnostics.json`.**
Artefact hashes and provenance: `outputs/manifest.json`.

---

## 1. The idea in one paragraph

NASA's GEDI instrument on the ISS fires a laser at the forest and measures canopy structure
directly, returning an above-ground biomass density estimate for a ~25 m footprint. It is the
closest thing to ground truth available at scale — but it samples along narrow orbital ground
tracks, not wall to wall. Sentinel-1 radar and Sentinel-2 optical imagery, by contrast, cover
everything, repeatedly, for free — but neither measures biomass. Arboris learns the mapping from
*what a given biomass looks like in radar and optical* at the places GEDI did measure, and applies
it where GEDI did not.

## 2. Data

| Source | Product | Role |
|---|---|---|
| NASA GEDI | L4A above-ground biomass density | **labels** (`agbd`, Mg/ha) |
| Copernicus Sentinel-1 | GRD, VV + VH, σ⁰ dB | radar structure/texture features |
| Copernicus Sentinel-2 | Surface reflectance, 10 bands | optical/spectral features |
| NASA SRTM | 30 m DEM | elevation, slope |

Pulled through Google Earth Engine. Result: `data/arboris_training.csv`, **27,773 GEDI
footprints** with co-located Sentinel-1/2 and terrain features.

### 2.1 Label quality filter

GEDI publishes its own per-shot standard error (`agbd_se`). Shots whose stated error exceeds
**50 % of the estimate itself** are dropped — the instrument is telling us it does not trust
them. **27,773 → 20,672 footprints survive (74.4 %).**

### 2.2 Aggregation to 200 m cells

Footprints are aggregated onto a 200 m grid (features averaged, labels averaged). Two reasons:
a single 25 m GEDI shot is noisy, and averaging several shots inside a cell suppresses that noise;
and it matches the effective resolution of the SAR texture features.

- **9,220 cells** contain at least one surviving footprint → these get predictions.
- **3,335 cells** contain **≥ 3** footprints → only these are trusted enough to train on.

### 2.3 Features (23)

Sentinel-2 bands `B2 B3 B4 B5 B6 B7 B8 B8A B11 B12`; indices `NDVI EVI SAVI NDWI NDRE`;
Sentinel-1 `VV VH VH_VV` and GLCM texture on VV (`contrast`, `entropy`, `variance`);
terrain `elevation`, `slope`.

Deliberately **excluded** from the feature matrix: `agbd` (the label), `agbd_se` (derived from the
label), and `lon`/`lat`. Feeding coordinates to a Random Forest lets it memorise *where* biomass is
instead of learning *what it looks like*, which produces an excellent score and a worthless model.

## 3. Model

Random Forest regressor, 500 trees, `min_samples_leaf=3`, `random_state=42`
(`src/data/train_model.py` → `models/arboris_rf.pkl`).

### 3.1 Validation — spatial, not random

Biomass is spatially autocorrelated: neighbouring cells are near-duplicates. A random train/test
split puts each cell's own neighbours in the test set and reports an accuracy that does not exist.

Arboris cuts the AOI into a **10 × 10 grid of blocks** and holds out **whole blocks** — 25 % of
them, repeated over 5 seeds. `src/diagnostics.py` asserts that **no block appears on both sides**
of the split and raises if one does.

## 4. Carbon accounting

Two multiplications, both IPCC defaults, stated out loud:

```
Mg CO₂e  =  AGB (Mg dry biomass)  ×  0.47  ×  44/12
                                     ↑        ↑
                    IPCC carbon fraction     molecular weight CO₂:C
                    of oven-dry biomass      = 3.6667
```

The two constants combine to **1.7233**. `carbon.sanity_check()` recomputes that ratio from the
actual output and raises `CarbonSanityError` if it drifts more than 1 %, because a violation means
a units bug (hectares vs km², m² vs hectares), never a model bug.

### 4.1 The area line that kills projects

```
cell_area_ha = (cell_width_m × cell_height_m) / 10_000
```

The grid is defined in *degrees*, so at latitude 10.38° a cell is 200.0 m tall but only
**196.7 m** wide. Actual cell area is **3.9346 ha**, not 4.00. Assuming a flat 4 ha would inflate
the headline tonnage by **1.66 %**.

### 4.2 Result

| | |
|---|---|
| Mapped area | **36,277 ha** (9,220 cells × 3.9346 ha) |
| Mean above-ground biomass | **132.1 Mg/ha** |
| Total above-ground biomass | **4,792,673 Mg** |
| Total carbon | **2,252,556 Mg C** |
| **Total CO₂e** | **8,259,373 Mg** |
| Mean CO₂e density | 227.7 Mg/ha |

## 5. Per-cell uncertainty

A Random Forest's 500 trees disagree more where the model is extrapolating. Arboris reports the
**standard deviation across the individual tree predictions** for every cell (mean 43.9 Mg/ha).

This is **model disagreement, not a calibrated confidence interval**, and it is labelled that way
in the UI. It earns its place because it was *validated*: see §3 of the validation report — the
least-confident third of held-out cells carries **2.37×** the RMSE of the most-confident third.

## 6. Stated limitations

1. **Coverage.** Predictions exist only for cells crossed by a GEDI ground track. Wall-to-wall
   output requires exporting the full Sentinel feature stack as a raster and running windowed
   inference over it — built but not yet run.
2. **Saturation.** Optical and C-band radar saturate in dense canopy. Above 300 Mg/ha the model
   under-predicts by ~123 Mg/ha. Quantified in the validation report rather than hidden.
3. **No radiometric slope correction** on Sentinel-1 yet. In steep terrain a hillside facing the
   satellite mimics high biomass, so the model partly learns topography. This is the most likely
   single cause of the modest R² and is the first thing to fix.
4. **Above-ground only.** Below-ground biomass is typically a further ~20–26 % of AGB in tropical
   forest; it is excluded here by choice, not by oversight.
5. **Carbon fraction.** 0.47 is a global default; species-specific values span ~0.45–0.50, so this
   constant contributes a few percent at most. The dominant error term is model RMSE, not this
   arithmetic.
6. **One epoch.** No change detection yet — that needs a second-date Sentinel export.

## 7. Reproducing

```bash
python src/data/train_model.py   # fit the model, write metrics.json + plots
python src/diagnostics.py        # hold-out diagnostics, uncertainty calibration
python src/predict_cells.py      # predict all cells, carbon totals, demo page
python src/manifest.py           # hash every artefact
python -m pytest tests/ -q       # 23 tests: carbon arithmetic, crown scoring, /api/crowns
```

`RANDOM_SEED = 42` throughout; reruns reproduce the same numbers.
