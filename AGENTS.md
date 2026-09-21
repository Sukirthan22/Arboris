# AGENTS.md — Arboris (ORION-PS-03)

**Repo:** `https://github.com/Sukirthan22/Arboris` · **Local:** `C:\Users\sukir\PROJECTS\SylvaSense`
**Read this file completely before writing a single line of code. It is the contract, not a suggestion.**

---

## 0. TL;DR for the agent

You are building **Arboris**: a satellite audit layer for carbon credits. Given a forest polygon (AOI),
it outputs (1) a biomass map in Mg/ha, (2) one headline tonnage of CO₂e, (3) a two-date change map,
(4) a tree-crown count on one high-resolution demo tile.

The system is one idea: **NASA's GEDI spaceborne laser already measured biomass at scattered ~25 m
footprints. We train a model to recognise what those footprints look like in free Sentinel-1 radar +
Sentinel-2 optical imagery, then predict biomass everywhere the laser never pointed.**

Three rules that override everything else in this document:

1. **Never fabricate a number.** Every figure that reaches the dashboard, the slides, or
   `docs/validation_report.md` must be traceable to a file in `outputs/` produced by a real run that
   is logged in `outputs/runs/`. If you cannot produce it, write `null` and say so loudly.
2. **Never silently fall back to synthetic data.** If Earth Engine auth fails, if a collection is
   empty, if a raster is misaligned — **raise**. A crash on day 1 is cheap; a demo built on fake
   pixels is fatal.
3. **Never widen scope.** §14 is a list of good ideas that are banned. If a change touches something
   on that list, stop and ask.

---

## 1. Project identity and non-negotiable invariants

| # | Invariant | Why it exists | Enforced by |
|---|---|---|---|
| I1 | GEDI footprints are the **only** source of AGB labels. No labels are invented, interpolated, or borrowed from another region. | Ground truth is the whole credibility argument. | `src/data/gedi.py`, `tests/test_gedi_filters.py` |
| I2 | Train/test split is **spatial (block-based)**, never random. | Neighbouring footprints are near-duplicates; a random split reports accuracy you don't have. | `src/validation/split.py`, `tests/test_split_leakage.py` |
| I3 | Every raster in the pipeline shares **one CRS, one transform, one shape** before any join. | Misaligned grids silently corrupt labels and cost hours of phantom debugging. | `src/io.py::assert_aligned` |
| I4 | Feature band order is defined **once** in `src/features/stack.py::FEATURE_ORDER`, persisted in model metadata, and re-asserted at predict time. | Silent column reordering between train and predict is the most common and most invisible ML bug in geospatial work. | `src/models/agb_rf.py`, `tests/test_feature_order.py` |
| I5 | Units are declared in every function name, docstring, and variable suffix (`_mg_ha`, `_ha`, `_m2`, `_mg_co2e`). | Hectares vs km² is the #1 cause of an absurd headline number. | Code review + `tests/test_carbon_math.py` |
| I6 | **No live model inference, no live Earth Engine call, during the demo.** All demo artefacts are precomputed static files. | A model that trains on stage is a model that fails on stage. | `api/main.py` reads only from `outputs/` |
| I7 | Crown counting is scoped to **one demo tile**, and every string in the UI and pitch says so. | "Per-tree across the whole AOI" is a lie a domain judge kills in one question. | UI copy review, §8.3 |
| I8 | `RANDOM_SEED = 42` is set globally; any run is reproducible from its config + seed. | "Run it again" must give the same number. | `src/config.py::set_global_seed` |

---

## 2. Repository map (authoritative)

Existing files are the skeleton. Files marked **NEW** you are to create. Do not invent directories
outside this tree without adding them here first.

```
Arboris/
├── AGENTS.md                      ← this file
├── README.md                      quickstart only; deep docs live in docs/
├── Makefile                       NEW  one-word entry points for every phase
├── environment.yml                NEW  conda/mamba env (AUTHORITATIVE on Windows)
├── requirements.txt               pip mirror, for Colab only
├── .env.example                   documented env vars, no secrets
├── .gitignore
│
├── config/
│   └── config.yaml                SINGLE SOURCE OF TRUTH for every tunable
│
├── src/
│   ├── __init__.py                NEW
│   ├── config.py                  NEW  typed config loader + seed + logging setup
│   ├── cli.py                     NEW  `python -m src.cli <phase>` entry points
│   ├── io.py                      raster/vector read-write, alignment assertions, COG writer
│   ├── data/
│   │   ├── gedi.py                Phase 0 — GEDI L4A pull + quality filtering
│   │   ├── sentinel.py            Phase 1 — S1 + S2 collection builders
│   │   └── preprocess.py          Phase 1 — speckle, terrain, cloud mask, compositing
│   ├── features/
│   │   └── stack.py               Phase 1 — FEATURE_ORDER, index math, GLCM, stack assembly
│   ├── validation/
│   │   └── split.py               NEW  spatial block split + leakage assertions
│   ├── models/
│   │   ├── agb_rf.py              Phase 2 — train/eval RF & XGB, persist model + metadata
│   │   ├── predict.py             NEW  windowed inference over the stack → biomass COG
│   │   └── crowns.py              Phase 3 — detectree2 fine-tune + inference + metrics
│   ├── carbon/
│   │   └── math.py                Phase 4 — AGB → C → CO₂e, with unit guards
│   └── alerts/
│       └── radd.py                Phase 5 — RADD import + two-date AGB differencing
│
├── api/
│   ├── main.py                    FastAPI app: static artefact serving + JSON endpoints
│   └── tiles.py                   rio-tiler/titiler raster tile router
│
├── web/                           NEW  thin frontend (single page, no build step if possible)
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── notebooks/
│   ├── 00_gedi_explore.ipynb      exploration ONLY — never imported by src/
│   └── 01_feature_sanity.ipynb    exploration ONLY
│
├── tests/                         NEW
│   ├── test_carbon_math.py
│   ├── test_feature_order.py
│   ├── test_split_leakage.py
│   ├── test_io_alignment.py
│   └── test_api_contract.py
│
├── data/                          gitignored
│   ├── raw/                       exports straight out of GEE
│   ├── interim/                   aligned intermediate rasters
│   └── processed/                 training table, feature stack COG
│
├── outputs/                       gitignored EXCEPT metrics.json + manifest.json
│   ├── agb_<aoi>_<date>.tif       biomass COG, float32, Mg/ha
│   ├── agb_diff_<t1>_<t2>.tif     degradation raster, float32, ΔMg/ha
│   ├── crowns_demo_tile.geojson   crown polygons
│   ├── radd_<aoi>.geojson         RADD alerts clipped to AOI
│   ├── metrics.json               EVERY number the UI displays
│   ├── manifest.json              artefact inventory + provenance + synthetic flags
│   ├── models/agb_rf_<runid>.joblib + .meta.json
│   └── runs/<runid>.json          full config snapshot + metrics + git SHA
│
└── docs/
    ├── methodology.md             the deliverable writeup (math + pipeline)
    ├── validation_report.md       R², RMSE, splits, feature importance, caveats
    └── demo_script.md             the golden path, word for word
```

**Hard rule:** `src/` never imports from `notebooks/`. `api/` never imports from `src/models/` at
request time — it reads files from `outputs/`. Notebooks may import from `src/`.

---

## 3. Environment

The geospatial stack (GDAL under rasterio/geopandas) is hostile to `pip` on Windows. **conda-forge is
authoritative.** `requirements.txt` exists only so Colab works.

`environment.yml`:

```yaml
name: arboris
channels: [conda-forge]
dependencies:
  - python=3.11
  - gdal
  - rasterio
  - rio-cogeo
  - geopandas
  - shapely
  - pyproj
  - numpy
  - pandas
  - scikit-learn
  - xgboost
  - matplotlib
  - pyyaml
  - pydantic>=2
  - pytest
  - jupyterlab
  - pip
  - pip:
      - earthengine-api
      - geemap
      - fastapi
      - uvicorn[standard]
      - titiler.core
      - rio-tiler
      - python-dotenv
```

Setup, once, and never deviate per-machine:

```bash
mamba env create -f environment.yml
conda activate arboris
earthengine authenticate          # opens browser, writes credentials to ~/.config/earthengine
python -c "import ee; ee.Initialize(project='<GEE_PROJECT_ID>'); print(ee.Number(1).getInfo())"
```

`.env.example` (copy to `.env`, never commit `.env`):

```
GEE_PROJECT_ID=your-ee-cloud-project
MAPBOX_TOKEN=pk.xxx            # only if Mapbox GL is used; deck.gl+MapLibre avoids this
ARBORIS_OUTPUT_DIR=./outputs
ARBORIS_LOG_LEVEL=INFO
```

**Lock the environment on day 1.** If a second person installs a different rasterio, you will spend
the night debugging a CRS difference, not a model.

---

## 4. Configuration contract

Everything tunable lives in `config/config.yaml`. **No magic numbers in code.** If you need a
constant, it goes here or into a named module-level constant with a comment citing its source.

```yaml
project:
  name: arboris
  run_id: null            # auto-filled at runtime: <YYYYMMDD-HHMMSS>-<git-sha7>
  seed: 42

aoi:
  name: <aoi_slug>              # e.g. nilgiris_north
  geojson_path: config/aoi/<aoi_slug>.geojson
  working_crs: EPSG:32643       # UTM zone containing the AOI. ALL rasters use this.
  area_crs: EPSG:6933           # equal-area (World Cylindrical Equal Area) for area math ONLY
  output_resolution_m: 20       # S2 20 m grid; S1 resampled up to match

dates:
  t1: ["2024-01-01", "2024-04-30"]   # dry-season window, epoch 1
  t2: ["2025-01-01", "2025-04-30"]   # epoch 2, for the change map
  gedi: ["2019-04-18", "2023-03-31"]

gedi:
  collection: LARSE/GEDI/GEDI04_A_002_MONTHLY
  quality_flag: 1
  degrade_flag: 0
  min_sensitivity: 0.95
  max_slope_deg: 20
  min_footprints_required: 500   # ABORT the AOI below this. Not a warning — an abort.

sentinel1:
  collection: COPERNICUS/S1_GRD
  instrument_mode: IW
  orbit_pass: DESCENDING         # pick ONE and never mix; incidence geometry differs
  polarisations: [VV, VH]
  speckle_filter: refined_lee    # refined_lee | focal_median
  speckle_kernel_m: 50
  slope_correction: volumetric   # Vollrath et al. 2020 angular/volumetric radiometric correction

sentinel2:
  collection: COPERNICUS/S2_SR_HARMONIZED
  cloud_prob_collection: COPERNICUS/S2_CLOUD_PROBABILITY
  max_cloud_prob: 40
  composite: median
  bands: [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]

terrain:
  dem: USGS/SRTMGL1_003

features:
  glcm_window: 3
  glcm_measures: [contrast, idm, ent]   # homogeneity == idm in GEE

model:
  kind: random_forest        # random_forest | xgboost
  random_forest:
    n_estimators: 500
    min_samples_leaf: 5
    max_features: sqrt
    n_jobs: -1
  xgboost:
    n_estimators: 800
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8

validation:
  strategy: spatial_block
  block_size_m: 2000         # >> spatial autocorrelation range of AGB
  test_fraction: 0.25
  min_test_footprints: 100

carbon:
  carbon_fraction: 0.47      # IPCC default carbon fraction of dry biomass
  co2_per_c: 3.6667          # 44/12 molecular weight ratio
  carbon_fraction_range: [0.45, 0.50]   # for the uncertainty answer, NOT for computation

crowns:
  model: detectree2
  benchmark: neon            # neon | selvabox
  demo_tile_path: data/raw/demo_tile.tif
  finetune_epochs: 30
  min_hand_labels: 40
  confidence_threshold: 0.5

alerts:
  radd_collection: projects/radar-wur/raddalert/v1
  radd_region: asia
  radd_min_confidence: 3     # RADD codes: 2 = low, 3 = high confidence
```

`src/config.py` loads this into a **pydantic v2 model**. Fail on unknown keys. Fail on missing keys.
No `dict.get(key, default)` scattered through the codebase — the default lives in the schema.

---

## 5. Data contracts (the schemas nothing may violate)

### 5.1 GEDI footprint table — `data/raw/gedi_<aoi>.parquet`

| column | dtype | unit | notes |
|---|---|---|---|
| `shot_id` | int64 | — | unique, primary key |
| `lon`, `lat` | float64 | deg | EPSG:4326 |
| `x`, `y` | float64 | m | `working_crs` |
| `agbd_mg_ha` | float32 | Mg/ha | **the label** |
| `agbd_se_mg_ha` | float32 | Mg/ha | GEDI's own standard error — keep it, quote it |
| `l4_quality_flag` | int8 | — | must == 1 |
| `degrade_flag` | int8 | — | must == 0 |
| `sensitivity` | float32 | — | must >= config |
| `slope_deg` | float32 | deg | from SRTM, must <= config |
| `acq_date` | datetime64 | — | |

### 5.2 Feature stack — `data/processed/stack_<aoi>_<epoch>.tif`

- **COG**, `float32`, nodata `-9999.0`, LZW-compressed, internally tiled 512×512, with overviews.
- CRS = `aoi.working_crs`. Resolution = `aoi.output_resolution_m`.
- Band order is **exactly** `FEATURE_ORDER` and band descriptions are written into the GeoTIFF:

```python
FEATURE_ORDER: tuple[str, ...] = (
    # optical indices
    "ndvi", "evi", "savi",
    # optical bands (surface reflectance, scaled 0-1)
    "b4_red", "b5_re1", "b6_re2", "b7_re3", "b8_nir", "b8a_nir_narrow", "b11_swir1", "b12_swir2",
    # SAR (dB)
    "vv_db", "vh_db", "vv_vh_ratio_db",
    # SAR texture (GLCM on VH)
    "vh_glcm_contrast", "vh_glcm_idm", "vh_glcm_ent",
    # terrain
    "elevation_m", "slope_deg",
)
```

Anything that adds, removes, or reorders a feature **must** bump `FEATURE_STACK_VERSION` and
invalidate cached models. `src/models/predict.py` asserts
`model.meta["feature_order"] == FEATURE_ORDER` and raises `FeatureOrderMismatch` otherwise.

### 5.3 Training table — `data/processed/training_<aoi>.parquet`

One row per GEDI footprint: all of §5.1 plus one column per name in `FEATURE_ORDER`, extracted as the
**mean of stack pixels within a 12.5 m radius buffer** of the footprint centre (GEDI footprint ≈ 25 m
diameter). Rows with any NaN feature are dropped and **the drop count is logged**. Plus:

| column | notes |
|---|---|
| `block_id` | spatial block index from `validation.block_size_m` |
| `split` | `"train"` / `"test"`, assigned by block, never by row |

### 5.4 Biomass raster — `outputs/agb_<aoi>_<epoch>.tif`

COG, `float32`, Mg/ha, nodata `-9999.0`, same grid as the stack. Negative predictions are **clipped to
0** (biomass cannot be negative) and the clipped-pixel fraction is recorded in `metrics.json`.

### 5.5 Crowns — `outputs/crowns_demo_tile.geojson`

EPSG:4326 FeatureCollection. Per feature: `crown_id` (int), `score` (float 0–1), `area_m2` (float).
Top-level `properties` on the collection: `tile_id`, `tile_bounds`, `n_crowns`, `model_version`,
`confidence_threshold`, `precision`, `recall`, `f1`, `benchmark`.

### 5.6 `outputs/metrics.json` — the only thing the UI is allowed to read numbers from

```json
{
  "run_id": "20260920-143122-a1b2c3d",
  "git_sha": "a1b2c3d",
  "generated_at": "2026-09-20T14:31:22Z",
  "aoi": {"name": "...", "area_ha": 12450.3, "crs": "EPSG:32643"},
  "model": {
    "kind": "random_forest",
    "n_train": 1842, "n_test": 613,
    "split": "spatial_block", "block_size_m": 2000,
    "r2": 0.64, "rmse_mg_ha": 46.2, "mae_mg_ha": 33.8, "bias_mg_ha": -1.4,
    "feature_importance": {"vh_db": 0.19, "ndvi": 0.15, "...": 0.0}
  },
  "biomass": {
    "mean_agbd_mg_ha": 148.7, "total_agb_mg": 1851360.0,
    "pct_pixels_clipped_to_zero": 0.4
  },
  "carbon": {
    "carbon_fraction": 0.47, "co2_per_c": 3.6667,
    "total_c_mg": 870139.2, "total_co2e_mg": 3190843.1,
    "mean_co2e_mg_ha": 256.3
  },
  "crowns": {
    "tile_id": "demo_01", "n_crowns": 1284,
    "precision": 0.71, "recall": 0.63, "f1": 0.67, "benchmark": "neon"
  },
  "change": {
    "t1": "2024-01/2024-04", "t2": "2025-01/2025-04",
    "net_agb_change_mg": -18420.0, "area_loss_ha": 62.1, "radd_alert_count": 47
  },
  "synthetic": false
}
```

`"synthetic": true` forces a **red banner across the top of the dashboard** reading
"DEMO DATA — NOT A REAL RUN". There is no way to display fake numbers without that banner. Do not add
one.

---

## 6. Module specifications

Every public function: full type hints, a docstring stating **units**, and structured logging. No
function silently swallows an exception. No bare `except:`.

### 6.1 `src/config.py`

```python
class Config(BaseModel):  # pydantic v2, model_config = ConfigDict(extra="forbid")
    ...

def load_config(path: str | Path = "config/config.yaml") -> Config: ...
def set_global_seed(seed: int) -> None:
    """Seed random, numpy, and PYTHONHASHSEED. Call at the top of every CLI entry point."""
def new_run_id() -> str:
    """<YYYYMMDD-HHMMSS>-<git sha7>; raises if the working tree is dirty and --allow-dirty is unset."""
def setup_logging(level: str) -> None:
    """stdlib logging, JSON lines to outputs/runs/<run_id>.log, human format to stderr."""
```

### 6.2 `src/io.py`

```python
def read_raster(path, band: int | None = None) -> tuple[np.ndarray, rasterio.Affine, CRS]: ...
def write_cog(path, array: np.ndarray, transform, crs, *, band_names: Sequence[str],
              nodata: float = -9999.0, dtype: str = "float32") -> Path:
    """Write a Cloud-Optimised GeoTIFF with band descriptions and overviews. rio-cogeo validated."""
def assert_aligned(*paths: str | Path) -> None:
    """Raise AlignmentError unless every raster shares CRS, transform, width, height.
    Call this before ANY pixel-wise join. This function is the cheapest bug insurance in the repo."""
def read_aoi(path) -> gpd.GeoDataFrame:
    """Load AOI, validate it is a single valid Polygon/MultiPolygon, reproject to working_crs."""
def aoi_area_ha(aoi: gpd.GeoDataFrame, area_crs: str) -> float:
    """Reproject to an EQUAL-AREA CRS before measuring. Never compute area in EPSG:4326."""
def write_manifest(outputs_dir: Path, entries: list[ArtefactEntry]) -> None: ...
```

### 6.3 `src/data/gedi.py` — Phase 0

```python
def init_ee(project_id: str) -> None:
    """ee.Initialize with an explicit project. Raise a readable error telling the user to run
    `earthengine authenticate` if credentials are missing. Never catch-and-continue."""

def fetch_gedi_footprints(aoi_ee: ee.Geometry, cfg: Config) -> gpd.GeoDataFrame:
    """Query GEDI L4A over the AOI and return quality-filtered footprints with AGBD in Mg/ha.

    Filters applied, in order (log the surviving count after EACH):
      1. date range
      2. l4_quality_flag == 1
      3. degrade_flag == 0
      4. sensitivity >= cfg.gedi.min_sensitivity
      5. SRTM slope <= cfg.gedi.max_slope_deg
    Raises InsufficientGroundTruth if the final count < cfg.gedi.min_footprints_required.
    """
```

Implementation notes the agent must honour:
- Export via `ee.batch.Export.table.toDrive` or `geemap.ee_to_gdf` for small AOIs. For anything
  large, **batch export and poll** — do not block on `getInfo()` and do not exceed the 5000-element
  client limit.
- Use `tileScale` 4–16 on reducers if you hit "computation timed out" / "user memory limit exceeded".
- Log the per-filter survival table into `outputs/runs/<run_id>.json`. When a judge asks how clean
  your labels are, that table *is* the answer.

**Done when:** `data/raw/gedi_<aoi>.parquet` exists, matches §5.1, and the count is printed and
logged.

### 6.4 `src/data/sentinel.py` + `src/data/preprocess.py` — Phase 1

```python
def s1_collection(aoi_ee, date_range, cfg) -> ee.ImageCollection:
    """COPERNICUS/S1_GRD, IW, single orbit pass, VV+VH present. GEE's S1_GRD is already
    thermal-noise-removed, calibrated to sigma0 and terrain-geocoded — what is NOT applied is
    radiometric slope correction, which we add ourselves in preprocess.apply_slope_correction."""

def s2_collection(aoi_ee, date_range, cfg) -> ee.ImageCollection:
    """S2_SR_HARMONIZED joined to S2_CLOUD_PROBABILITY on system:index; mask where
    probability > cfg.sentinel2.max_cloud_prob; also mask SCL classes 3,8,9,10 (shadow/cloud/cirrus)."""

def speckle_filter(img: ee.Image, cfg) -> ee.Image:
    """Refined Lee (or focal median fallback). Operates in LINEAR power, not dB.
    Filtering in dB is wrong — log-domain averaging biases the mean low."""

def apply_slope_correction(img: ee.Image, dem: ee.Image) -> ee.Image:
    """Volumetric radiometric slope correction (Vollrath et al. 2020). Without it, a hillside
    facing the satellite fakes high biomass and your model learns topography, not trees."""

def median_composite(coll: ee.ImageCollection, bands: Sequence[str]) -> ee.Image: ...
def to_db(img: ee.Image, bands) -> ee.Image:  # 10*log10, AFTER speckle filtering
```

Order of operations is fixed and non-negotiable:
`filter collection → mask clouds (S2) → speckle filter in linear (S1) → slope correction (S1) →
convert to dB (S1) → median composite → compute indices → GLCM → add terrain → clip to AOI → reproject
to working_crs at output_resolution_m`.

### 6.5 `src/features/stack.py` — Phase 1

```python
FEATURE_ORDER: tuple[str, ...]          # §5.2 — the single source of truth
FEATURE_STACK_VERSION: int = 1

def compute_indices(s2: ee.Image) -> ee.Image:
    """NDVI=(NIR-R)/(NIR+R); EVI=2.5*(NIR-R)/(NIR+6R-7.5B+1); SAVI=1.5*(NIR-R)/(NIR+R+0.5).
    S2 SR is scaled by 1e4 — divide before computing EVI or the soil/aerosol terms are nonsense."""

def glcm_texture(s1_db: ee.Image, cfg) -> ee.Image:
    """GEE glcmTexture requires an INTEGER band. Rescale VH dB to 0-255 uint8 over a FIXED
    range (e.g. -30..0 dB) — never a per-image min/max, which makes texture non-comparable
    across dates and breaks the change map."""

def build_stack(aoi_ee, epoch: str, cfg) -> ee.Image:
    """Assemble and return an ee.Image with bands named and ordered exactly as FEATURE_ORDER."""

def export_stack(img: ee.Image, aoi_ee, cfg, out_path: Path) -> Path: ...

def sample_footprints(stack_path: Path, gedi: gpd.GeoDataFrame, cfg) -> pd.DataFrame:
    """Buffer each footprint by 12.5 m, take the mean of each band inside, join to the label.
    Runs LOCALLY against the exported COG via rasterio (rasterstats-style), not in GEE —
    this keeps the join auditable and keeps you off the GEE element limit."""
```

### 6.6 `src/validation/split.py`

```python
def assign_blocks(df: pd.DataFrame, block_size_m: float, crs: str) -> pd.Series:
    """Floor x/y into a regular grid: block_id = f"{int(x//s)}_{int(y//s)}"."""

def spatial_split(df, cfg, seed: int) -> pd.DataFrame:
    """Shuffle BLOCKS (not rows) and assign whole blocks to test until test_fraction is reached.
    Raise if the resulting test set has fewer than cfg.validation.min_test_footprints rows."""

def assert_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any block_id appears in both splits, or if any train point is within
    block_size_m/2 of a test point. Called unconditionally inside train()."""
```

### 6.7 `src/models/agb_rf.py` — Phase 2

```python
@dataclass(frozen=True)
class TrainResult:
    model_path: Path; meta_path: Path; metrics: dict[str, float]

def train(training_table: Path, cfg: Config, run_id: str) -> TrainResult:
    """Fit RF or XGB on FEATURE_ORDER columns → agbd_mg_ha. Evaluate ONLY on split=='test'.
    Reports r2, rmse_mg_ha, mae_mg_ha, bias_mg_ha, plus a permutation feature importance.
    Persists joblib + a .meta.json containing: feature_order, FEATURE_STACK_VERSION, config hash,
    git sha, seed, n_train, n_test, sklearn version, metrics."""

def evaluate(model, X, y) -> dict[str, float]: ...
def plot_diagnostics(y_true, y_pred, importances, out_dir: Path) -> list[Path]:
    """Three plots, all of which end up in the deck: predicted-vs-observed with 1:1 line,
    residuals vs predicted (shows saturation at high biomass — a known and expected effect,
    say so out loud rather than hiding it), and a horizontal feature-importance bar chart."""
```

**Sanity gate:** if `r2 > 0.90`, `train()` logs a **prominent warning** and writes
`"suspicious_r2": true` into metrics. The three usual causes, in likelihood order: (1) the split was
effectively random, (2) a label leaked into the feature table, (3) the test set is tiny. Investigate
before it reaches a slide. Published work in this space is R² 0.6–0.85 / RMSE 40–55 Mg/ha.

### 6.8 `src/models/predict.py`

```python
def predict_raster(stack_path: Path, model_path: Path, out_path: Path, *,
                   block_size: int = 512) -> Path:
    """Windowed inference: rasterio block-by-block, never load the whole stack into RAM.
    - assert model.meta['feature_order'] == FEATURE_ORDER  (else FeatureOrderMismatch)
    - mask nodata pixels BEFORE predicting; write nodata back out, do not predict on -9999
    - clip negative predictions to 0 and count them
    - write via io.write_cog
    """
```

### 6.9 `src/carbon/math.py` — Phase 4

The whole deliverable is two multiplications. Make them unimpeachable.

```python
CARBON_FRACTION: Final[float] = 0.47      # IPCC default carbon fraction of dry biomass
CO2_PER_C: Final[float] = 44.0 / 12.0     # 3.6667, molecular weight ratio CO2:C

def agb_to_carbon(agb_mg_ha: float | np.ndarray, fraction: float = CARBON_FRACTION): 
    """Mg dry biomass/ha → Mg C/ha. Raises on negative input."""

def carbon_to_co2e(c_mg_ha, ratio: float = CO2_PER_C):
    """Mg C/ha → Mg CO2e/ha."""

def raster_to_total_co2e(agb_raster_path: Path, area_crs: str) -> CarbonSummary:
    """Per-pixel Mg/ha → total Mg CO2e for the AOI.
    pixel_area_ha = (pixel_width_m * pixel_height_m) / 10_000   ← the line that kills projects
    total = sum(valid_pixels * pixel_area_ha) * 0.47 * 3.6667
    Returns mean_agbd_mg_ha, area_ha, total_agb_mg, total_c_mg, total_co2e_mg, n_valid_pixels."""

def sanity_check(mean_agbd_mg_ha: float, mean_co2e_mg_ha: float) -> None:
    """The two constants combine to ~1.7233. Raise CarbonSanityError if the observed ratio
    deviates >1%. A forest at 150 Mg/ha must land near 258 Mg CO2e/ha. If it doesn't, you have
    a unit bug (hectares vs km², or m² vs hectares), not a model bug."""
```

`tests/test_carbon_math.py` must assert: `150 → 70.5 Mg C/ha → 258.5 Mg CO₂e/ha` (±0.1), that
negatives raise, that NaN propagates rather than silently becoming 0, and that a 1 km² AOI at
100 Mg/ha totals 10,000 Mg AGB (not 1,000,000).

**Uncertainty line for Q&A (put it in `docs/methodology.md`):** 0.47 is a global default;
species-specific carbon fractions range ~0.45–0.50, so the constant contributes a few percent at
most. The dominant error is the Phase 2 model RMSE, not this arithmetic.

### 6.10 `src/models/crowns.py` — Phase 3

```python
def load_pretrained(cfg) -> "detectree2.Model":
    """detectree2 (Detectron2 / Mask R-CNN), tropical-canopy weights. Never train from scratch."""

def finetune(tile_path: Path, labels_geojson: Path, cfg) -> Path:
    """Fine-tune on >= cfg.crowns.min_hand_labels hand-drawn crowns from OUR tile.
    Cross-biome generalisation is the known failure mode: published average precision collapsed
    from 0.670 on the training forest type to 0.094 on a different one. Running pretrained weights
    blind on an unfamiliar forest is the single most likely way this phase embarrasses you on stage."""

def predict_crowns(tile_path: Path, model, cfg) -> gpd.GeoDataFrame:
    """Tile the image with overlap, run inference, then de-duplicate overlapping crowns across
    tile seams (IoU-based NMS at ~0.4) before counting. Forgetting the seam dedup inflates the
    tree count and is trivially caught by a judge zooming into a boundary."""

def evaluate_crowns(pred: gpd.GeoDataFrame, truth: gpd.GeoDataFrame,
                    iou_threshold: float = 0.5) -> dict[str, float]:
    """Greedy IoU matching → precision, recall, F1. Evaluated against NeonTreeEvaluation or
    SelvaBox labels so the numbers are MEASURED, not asserted."""
```

Every UI string and pitch line for this phase reads **"per-tree resolution on the demo tile."** Never
"across the AOI."

### 6.11 `src/alerts/radd.py` — Phase 5

```python
def fetch_radd(aoi_ee, cfg) -> gpd.GeoDataFrame:
    """projects/radar-wur/raddalert/v1, band 'alert' (confidence) + 'date' (YYDOY encoded).
    Decode dates properly: yy = floor(v/1000), doy = v % 1000. Filter to the t1..t2 window and to
    confidence >= cfg.alerts.radd_min_confidence. Vectorise to polygons for the map layer."""

def agb_difference(t1_raster: Path, t2_raster: Path, out_path: Path) -> ChangeSummary:
    """assert_aligned first, ALWAYS. Δ = t2 - t1 in Mg/ha. Report net change, gross loss,
    area with loss > 1 RMSE (anything smaller is noise, not signal — say this when asked)."""
```

**The differentiation argument, which belongs in both the code comments and the pitch:** RADD detects
*clearing* — binary, "trees were removed here on this date." Our AGB difference detects *degradation*
— continuous, "this area lost 30 Mg/ha without being clear-cut": selective logging, thinning, fire
damage. RADD does not catch that, and that is exactly the failure mode that makes carbon credits
fraudulent. That contrast is the answer to "why not just use existing tools?"

### 6.12 `api/main.py` + `api/tiles.py` — Phase 6

FastAPI. **Reads only from `outputs/`.** Never imports a model. Never calls Earth Engine.

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | `{"status":"ok","run_id":...,"artefacts_present":[...]}` |
| GET | `/api/aoi` | AOI GeoJSON (EPSG:4326) |
| GET | `/api/summary` | the `outputs/metrics.json` blob, verbatim |
| GET | `/api/crowns` | crowns GeoJSON |
| GET | `/api/alerts/radd` | RADD GeoJSON |
| GET | `/api/manifest` | artefact inventory + provenance |
| GET | `/tiles/{layer}/{z}/{x}/{y}.png` | rio-tiler PNG from the COG; `layer ∈ {agb, agb_diff}` |
| GET | `/tiles/{layer}/tilejson.json` | TileJSON for the frontend |

Rules:
- Startup validates that every artefact in `manifest.json` exists on disk; log a loud warning per
  missing one, and expose it through `/api/health`. A blank layer at demo time must be diagnosable
  in five seconds.
- Colormaps and rescale ranges are **fixed in config**, not auto-scaled per tile — otherwise the
  legend lies and the two epochs aren't comparable.
- CORS open for localhost only.
- No auth, no database, no user accounts (§14).

### 6.13 `web/` — the frontend

MapLibre GL (or Mapbox GL if a token exists) + deck.gl. One page. No framework build step unless it
is genuinely free.

Minimum viable screen, nothing more:
- Basemap with the AOI outlined.
- Layer toggles: **biomass heatmap · tree crowns · RADD alerts · degradation difference.**
- One big number panel: total Mg CO₂e, area in hectares, tree count *on the demo tile*.
- A small panel showing **R² and RMSE**. Displaying your own error is a credibility move, not a
  weakness — put the error bar on the slide yourself, before a judge asks for it.
- A legend with fixed units for every raster layer.

This is a display shell around Phases 2–5. It is the easiest place in the project to lose a whole day
to CSS. Do not.

---

## 7. CLI and Makefile contract

```bash
python -m src.cli phase0   --config config/config.yaml        # GEDI pull
python -m src.cli phase1   --config ... [--epoch t1|t2]       # stack + sample
python -m src.cli phase2   --config ...                       # train + predict raster
python -m src.cli carbon   --config ...                       # totals → metrics.json
python -m src.cli phase3   --config ... [--finetune]          # crowns
python -m src.cli phase5   --config ...                       # RADD + AGB difference
python -m src.cli metrics  --config ...                       # assemble metrics.json + manifest.json
python -m src.cli serve    --config ... [--port 8000]
```

Every subcommand: `--dry-run` (validate config and inputs, touch nothing), `--force` (recompute
instead of reusing cached artefacts), and idempotency — rerunning must not corrupt prior outputs.
Every subcommand writes `outputs/runs/<run_id>.json` with the resolved config, git SHA, wall time,
input hashes, and the metrics it produced.

`Makefile`: `make env`, `make phase0`, … `make all`, `make test`, `make demo` (serve + open browser).

---

## 8. Definition of done — the gates

No phase is "done" by feeling done. Each gate is a file on disk plus a printed number.

| Phase | Done when |
|---|---|
| **0** | `data/raw/gedi_<aoi>.parquet` exists, quality-filtered per §5.1, count ≥ `min_footprints_required`, per-filter survival table logged. |
| **1** | `data/processed/stack_<aoi>_t1.tif` passes `assert_aligned` and a COG validity check; `training_<aoi>.parquet` has one row per surviving footprint with every `FEATURE_ORDER` column non-null; NaN-drop count logged. |
| **2** | `outputs/agb_<aoi>_t1.tif` covers the full AOI; `metrics.json` carries R², RMSE, MAE, bias from a **spatial** hold-out with `assert_no_leakage` passing; three diagnostic plots saved. |
| **3** | `outputs/crowns_demo_tile.geojson` with polygons + count for one curated tile, precision/recall/F1 measured against benchmark labels, seam-dedup applied. |
| **4** | `carbon` block in `metrics.json` populated; `sanity_check` passes; `docs/methodology.md` states both constants with units and sources. |
| **5** | RADD layer renders as a toggle over the AOI; `outputs/agb_diff_t1_t2.tif` renders red/green with a total biomass-change figure. |
| **6** | A stranger opens the URL, toggles every layer, and reads the carbon number **without you touching the keyboard.** |
| **7** | Golden path (§12) run end-to-end three times on the demo machine, on the demo network, with a screen recording saved as fallback. |

---

## 9. Testing requirements

`pytest tests/ -q` must pass before any commit that touches `src/`.

- `test_carbon_math.py` — the fixed-value assertions in §6.9. These are exact; they never change.
- `test_feature_order.py` — `FEATURE_ORDER` has no duplicates; a model's metadata order must equal
  the current order; a deliberately shuffled order raises `FeatureOrderMismatch`.
- `test_split_leakage.py` — synthetic clustered points; assert no `block_id` spans both splits;
  assert a random split would have been rejected.
- `test_io_alignment.py` — two rasters with different transforms raise `AlignmentError`.
- `test_api_contract.py` — every endpoint returns the documented shape against a fixture
  `outputs/` directory; `/api/summary` keys match §5.6 exactly.

Tests must not require Earth Engine credentials. GEE-touching functions take an injected client or are
excluded via `@pytest.mark.network` (deselected by default).

---

## 10. Logging, provenance, error handling

- **Structured logging.** One line per stage: `stage`, `run_id`, `duration_s`, `n_in`, `n_out`, and
  the key metric. Human-readable to stderr, JSON lines to `outputs/runs/<run_id>.log`.
- **Named exceptions**, all in `src/exceptions.py`: `InsufficientGroundTruth`, `AlignmentError`,
  `FeatureOrderMismatch`, `CarbonSanityError`, `ArtefactMissing`, `EarthEngineAuthError`. Each message
  states what was expected, what was found, and the single next action.
- **No silent defaults.** Missing config key → raise. Empty collection → raise. Zero footprints →
  raise. The "log a warning and continue" pattern is how you end up presenting a map of nothing.
- **Provenance.** Every output file has an entry in `manifest.json`: path, sha256, run_id, git SHA,
  source config hash, generation timestamp, `synthetic: bool`.

---

## 11. Git and change discipline

- Branches: `phase/<n>-<slug>`. Commits: Conventional Commits (`feat(phase2): spatial block split`).
- One phase per PR-sized change. Never refactor and add a feature in the same commit.
- `data/` and `outputs/` are gitignored **except** `outputs/metrics.json` and `outputs/manifest.json`,
  which are committed so the numbers in the deck are version-controlled.
- Never commit `.env`, GEE credentials, Mapbox tokens, or `.joblib` model files.
- Before any commit that changes a number in the deck: rerun `make test` and regenerate
  `metrics.json`. A slide number without a matching committed `metrics.json` is not allowed to exist.

---

## 12. The golden path (what the software exists to do)

The build is judged on whether the biomass number is defensible and the demo runs. This is the
sequence, rehearsed until it needs no thinking:

1. **Pick the AOI** — map opens on our forest, boundary already drawn.
2. **Fused biomass map** — toggle the heatmap. One sentence: radar + optical + NASA laser labels.
3. **The carbon number** — headline tonnage of CO₂e, both constants named out loud.
4. **The demo tile** — zoom in, crowns outlined, tree count on screen. Say *"per-tree resolution on
   this tile."*
5. **Deforestation overlay** — RADD alerts, then the degradation difference. Contrast clearing vs
   degradation. **Then stop talking.**

The three questions you will definitely be asked, and where the answers live:
- *How do you know your biomass number is right?* → spatial hold-out against GEDI, R² and RMSE, and
  it's on the slide (`metrics.json` → validation panel).
- *Why not just use existing tools?* → degradation vs clearing, per-plot self-serve (§6.11).
- *Does the government accept this method?* → the Forest Survey of India, with ISRO's Space
  Applications Centre, already runs pan-India AGB estimation from SAR. We're making the government's
  own method self-serve and auditable.

---

## 13. Failure modes, ranked by how likely they are to ruin the demo

| # | Symptom | Real cause | Fix |
|---|---|---|---|
| 1 | R² suspiciously high (>0.9) | random split, label leak, or tiny test set | `assert_no_leakage`; inspect `FEATURE_ORDER` for a label-derived column; check `n_test` |
| 2 | Headline CO₂e absurd by 10²–10⁶ | hectares vs km², or m² vs hectares | `carbon.sanity_check`; ratio must be ≈1.7233 |
| 3 | Blank/garbage biomass map | nodata predicted on, or CRS mismatch | mask before predict; `assert_aligned` |
| 4 | Model learns topography | no radiometric slope correction on S1 | `apply_slope_correction` |
| 5 | Tree count inflated | no seam dedup across inference tiles | IoU NMS at ~0.4 |
| 6 | Crown detection collapses on our forest | cross-biome generalisation failure (AP 0.670 → 0.094) | hand-label ≥40 crowns, fine-tune |
| 7 | GEE "user memory limit exceeded" | reducer over too large a region | raise `tileScale`, batch export, tile the AOI |
| 8 | Nothing loads on stage | live inference / live GEE call | precompute; static files only (I6) |
| 9 | `pip install rasterio` explodes | GDAL on Windows | conda-forge, or Colab |
| 10 | Two epochs not comparable | per-image min/max scaling in GLCM or tile colormap | fixed ranges in config |

---

## 14. Out of scope — refuse these

Every item below is a reasonable idea that will not survive contact with the clock. If a request
touches one, stop and point here.

- Per-tree counting across the whole AOI (needs sub-metre imagery we don't have). **Demo tile only.**
- Building our own deforestation detector. RADD exists, is free, and is better than a one-day build.
- Below-ground biomass, soil carbon, dead wood. Say **"above-ground only"** in the pitch and it
  becomes a stated scope rather than an oversight.
- Species identification.
- Training any model from scratch. Pretrained + fine-tune, always.
- User accounts, auth, multi-tenancy, a database. Static files and one AOI.
- Live inference in the browser.
- A second AOI. One AOI done properly beats two half-processed.

**The governing rule:** this project is judged on whether the biomass number is defensible and the
demo runs. Nothing on that list moves either.

---

## 15. How the coding agent behaves

1. **Read `config/config.yaml` before touching any module.** Never hardcode a path, date, band, CRS,
   or threshold that config already owns.
2. **Work one module at a time**, smallest diff that satisfies the spec in §6. State which §
   you're implementing.
3. **Run `pytest` after every change to `src/`.** Report pass/fail, don't assume.
4. **When a spec here conflicts with what you'd naturally do, this file wins.** If it's genuinely
   wrong, say so explicitly and propose the edit — don't quietly deviate.
5. **When information is missing** (AOI not chosen, demo tile not acquired, GEE project unknown),
   stop and ask one specific question. Do not invent a plausible value.
6. **Every number you report** cites the file it came from. "R² is 0.64" is incomplete;
   "R² 0.64 from `outputs/runs/20260920-143122-a1b2c3d.json`" is complete.
7. **Never mark a phase done** without the §8 gate artefact existing on disk.
8. **If you write placeholder data for UI development**, set `"synthetic": true` in `metrics.json`
   and leave the red banner intact. Removing the banner is a breaking change.
9. **No new dependency** without adding it to `environment.yml` and stating why the stdlib or an
   existing dependency doesn't cover it.
10. **Honesty over polish.** A stated R² of 0.64 with a spatial hold-out and a named RMSE reads as a
    team that knows what it built. A suspiciously perfect number reads as a bug or a lie, and either
    way the judge stops believing the rest of the deck.
