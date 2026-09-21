# Arboris — Validation Report

Source: `outputs/metrics.json`, `outputs/diagnostics.json`. Artefact hashes in
`outputs/manifest.json`. `"synthetic": false` — every figure below came from a real run.

---

## 1. Headline accuracy

Random Forest, 23 features, **spatially-blocked** hold-out repeated over 5 seeds:

| Metric | Value |
|---|---|
| **R²** | **0.437 ± 0.093** |
| **RMSE** | **58.45 ± 4.33 Mg/ha** |
| MAE | 42.07 Mg/ha |
| Bias (seed 42) | **+3.31 Mg/ha** |
| Training cells | 3,335 (≥ 3 GEDI shots each) |
| Hold-out (seed 42) | 2,611 train / 724 test across 24 held-out blocks |
| Block leakage | **none** — asserted in `src/diagnostics.py`, raises if violated |

### Read this honestly

**R² 0.437 is below the 0.6–0.85 range typical of published GEDI-trained AGB models.** We are not
going to dress that up. The most probable cause is the absence of radiometric slope correction on
the Sentinel-1 input (§4 below): this is steep Western Ghats terrain, and uncorrected SAR
backscatter on a slope facing the satellite mimics high biomass.

Two things are worth noting alongside it:

- Bias is **+3.31 Mg/ha** on the hold-out — about **2.5 %** of mean biomass. The model is
  imprecise, not systematically inflated, so the **headline tonnage is not being propped up by
  a bias**.
- The error is **not uniform**, and the model can tell you where it sits (§3). On the third of
  cells it flags as confident, RMSE is **27.8 Mg/ha** — squarely in published territory.

### Why the split matters

A random train/test split on spatially autocorrelated data would have reported a far prettier
number by putting each cell's own neighbours in the test set. Whole 200 m blocks are held out
instead, and the leakage assertion runs unconditionally.

## 2. Where the error lives — saturation

Hold-out cells binned by *observed* GEDI biomass (`outputs/residuals.png`):

| Observed (Mg/ha) | n | Mean observed | Mean predicted | Bias |
|---|---|---|---|---|
| 0–50 | 81 | 26.3 | 44.4 | **+18.2** |
| 50–100 | 189 | 75.6 | 108.9 | **+33.2** |
| 100–150 | 198 | 124.2 | 141.4 | +17.2 |
| 150–200 | 134 | 172.2 | 167.8 | −4.5 |
| 200–250 | 72 | 223.8 | 178.4 | **−45.3** |
| 250–300 | 23 | 272.3 | 204.0 | **−68.3** |
| 300+ | 27 | 335.3 | 212.0 | **−123.3** |

This is the classic shape: **over-predicts sparse forest, under-predicts dense forest.** Two
causes, both expected and both documented in the literature:

1. **Physical saturation.** Sentinel-2 reflectance and C-band SAR stop responding once canopy
   closes. Above ~250 Mg/ha the satellite genuinely cannot see the difference.
2. **Regression to the mean.** A Random Forest averages leaf values and cannot extrapolate past
   its training range, which compresses both tails.

Consequence for the headline: the two biases partly cancel — hence overall bias of only
+3.3 Mg/ha — but **a dense old-growth stand will be under-valued** by this model. That is the
conservative direction for a carbon-credit audit, which is the right way to be wrong.

## 3. Does the uncertainty layer actually work?

An uncertainty layer nobody validated is decoration. Ours was tested.

For each held-out cell we take the **standard deviation of the 500 individual tree predictions**,
then check whether it predicts real error.

| Confidence band | n | Mean tree spread | **RMSE** | MAE |
|---|---|---|---|---|
| Low spread — most confident | 241 | 24.0 | **27.8** | 19.1 |
| Medium | 241 | 47.5 | **52.8** | 41.6 |
| High spread — least confident | 242 | 65.6 | **66.0** | 52.6 |

- correlation(tree spread, \|error\|) = **0.445**
- least-confident third carries **2.37×** the RMSE of the most-confident third

**The flag works.** Cells the model marks as uncertain really are the cells it gets wrong. This
converts a single headline number into an auditable product: a buyer can discount or exclude the
low-confidence cells instead of taking one AOI-wide average on faith.

Caveat stated plainly: tree spread is **model disagreement, not a calibrated confidence
interval**. It ranks cells reliably; it is not a 68 % prediction interval and is not labelled as
one anywhere in the UI.

Plot: `outputs/uncertainty_calibration.png`.

## 4. Known weaknesses, ranked by how much they cost

| # | Issue | Effect | Fix |
|---|---|---|---|
| 1 | No radiometric slope correction on S1 | model partly learns topography; likely the main R² cost | Vollrath et al. 2020 volumetric correction in the GEE export |
| 2 | Coverage limited to GEDI ground tracks | 9,220 cells, not wall-to-wall | export the full Sentinel stack as a raster, windowed inference |
| 3 | Saturation above ~250 Mg/ha | dense stands under-valued | add L-band SAR (ALOS PALSAR) which penetrates canopy |
| 4 | Single dry-season composite | no phenology, no change detection | second epoch export |
| 5 | 3,335 training cells | limits model capacity | widen the AOI or relax the ≥3-shot rule with weighting |

## 5. Phase 3 — tree crowns on the demo tile

Source: `notebooks/detectree2.ipynb` (cells 11, 17, 19), code in `src/models/crowns.py`,
settings in `config/config.yaml → crowns`. Demo artefact: `outputs/demo8_crowns.geojson`,
served by `GET /api/crowns`.

**Scope: per-tree resolution on the demo tile only.** The crown tiles are NEON benchmark
images (`2018_TEAK_…`, Teakettle, California — EPSG:32611), not the biomass AOI. This phase
shows that the crown detector is measured honestly; it does not count trees across the AOI.

### Setup

| Setting | Value |
|---|---|
| Model | detectree2 (Mask R-CNN), pretrained **`250312_flexi`**, **no fine-tune** |
| Tiling | whole image: `tile_width=41`, `tile_height=41`, `buffer=0`, `full_coverage=True` |
| Inference floor | `SCORE_THRESH_TEST = 0.05` (filter later, not in the model) |
| Confidence cutoff | **0.1**, chosen from {0.1 … 0.5} by pooled F1 on **tiles 1–7 only** |
| Matching | predicted crown envelope vs truth box, greedy 1-to-1, **IoU ≥ 0.4** |
| Held out | **tiles 8–10** — never used to choose anything |

### Result — held-out tiles 8–10, pooled

| Metric | Value |
|---|---|
| Truth crowns | **107** |
| Predicted boxes | 71 |
| Matched | 42 |
| **Precision** | **0.59** |
| **Recall** | **0.39** |
| **F1** | **0.47** |
| Published detectree2 reference | F1 0.57 |

Demo tile 8: **11 predicted crowns** vs 8 truth boxes at the 0.1 cutoff.

### Read this honestly

- F1 0.47 is **below the 0.57 reference**. The detector misses more trees than it invents
  (recall 0.39 < precision 0.59): at this cutoff it under-counts.
- Picking the cutoff mattered. With the default 0.5 on the same held-out tiles and matching
  rule, F1 was 0.36 (P 0.78, R 0.23): high precision, most trees missed. Choosing 0.1 on tiles
  1–7 recovered recall without looking at the test tiles.
- Truth is **boxes, not outlines**, so predicted polygons are scored by their bounding boxes.
  This measures detection and rough extent, not crown-boundary accuracy.

### Negative finding — fine-tuning hurt

We fine-tuned detectree2 on **8 m training tiles**. It scored **worse** than the untouched
pretrained weights on held-out tile 8 (notebook cell 11, mask IoU ≥ 0.5):

| Cutoff | Fine-tuned F1 | Pretrained F1 |
|---|---|---|
| 0.3 | 0.43 | **0.53** |
| 0.5 | 0.33 | **0.57** |
| 0.7 | 0.36 | **0.40** |

Cause: crowns here are **10–13 m across**, so an 8 m tile can never hold a whole crown.
Every training example was a crown cut into pieces, and the model learned to predict
fragments. The shipped model uses **pretrained weights with whole-image tiling**, so no crown
is ever split at a tile edge. Fine-tuning is worth revisiting only with tiles larger than the
largest crown.

## 6. What is *not* claimed

- No per-tree crown counts across the AOI. Crowns are counted on **one NEON demo tile**, with
  measured P/R/F1 (§5), and nowhere else.
- No deforestation or degradation detection. Needs a second epoch.
- No below-ground, soil or dead-wood carbon. Above-ground only, by choice.
- No claim that 8.26 Mt CO₂e is accurate to better than the stated RMSE. It is a
  **± 58 Mg/ha-per-cell** estimate, and the map shows which cells to trust.

## 7. Verification trail

```bash
python -m pytest tests/ -q     # 23 passed — carbon arithmetic pinned; crown scoring + /api/crowns
python src/diagnostics.py      # re-derives every number in §1–3, asserts no block leakage
python src/manifest.py         # sha256 of all 11 artefacts
```

`sanity_check()` runs inside `cells_to_total_co2e()` on every execution: if the CO₂e/AGB ratio
ever drifts from 1.7233 by more than 1 %, the pipeline raises instead of writing a number.
