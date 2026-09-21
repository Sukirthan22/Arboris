"""
Arboris — Phase 2: AGB regression model
Run:  python train_model.py

Input :  data/arboris_training.csv   (GEDI labels + Sentinel-1/2 + terrain)
Output:  models/arboris_rf.pkl       trained model
         outputs/metrics.json        the numbers for your deck
         outputs/scatter.png         predicted vs actual plot (slide 'validation')
         outputs/feature_importance.png
"""
import json, warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- config
CSV       = "data/arboris_training.csv"
CELL_M    = 200      # aggregate GEDI footprints into 200 m cells
MIN_SHOTS = 3        # a cell needs >= 3 laser shots to be trusted
N_BLOCKS  = 10       # AOI is cut into a 10 x 10 grid for the split
TEST_FRAC = 0.25     # 25% of blocks held out
SEEDS     = [42, 7, 101, 2024, 55]   # repeat the split 5x, report mean +/- sd

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# ---------------------------------------------------------------- load
df = pd.read_csv(CSV)
coords = df[".geo"].map(lambda s: json.loads(s)["coordinates"])
df["lon"] = [c[0] for c in coords]
df["lat"] = [c[1] for c in coords]
print(f"loaded {len(df):,} GEDI footprints")

# drop shots GEDI itself flags as unreliable (error > 50% of the estimate)
df = df[(df.agbd_se / df.agbd) <= 0.5].copy()
print(f"{len(df):,} left after quality filter")

# these must never become model inputs:
#   system:index = row id, .geo = geometry, agbd = the answer,
#   agbd_se = uncertainty derived FROM the answer, lon/lat = location
DROP = ["system:index", ".geo", "agbd", "agbd_se", "lon", "lat"]
FEATURES = [c for c in df.columns if c not in DROP]
print(f"{len(FEATURES)} features: {FEATURES}")

# ---------------------------------------------------------------- aggregate
deg = CELL_M / 111_320.0
df["cx"] = np.floor(df.lon / deg).astype(int)
df["cy"] = np.floor(df.lat / deg).astype(int)

agg = (df.groupby(["cx", "cy"])
         .agg(**{f: (f, "mean") for f in FEATURES},
              agbd=("agbd", "mean"),
              lon=("lon", "mean"),
              lat=("lat", "mean"),
              nshot=("agbd", "size"))
         .reset_index())
agg = agg[agg.nshot >= MIN_SHOTS].reset_index(drop=True)
print(f"{len(agg):,} cells at {CELL_M} m with >= {MIN_SHOTS} shots")

agg["block"] = (pd.cut(agg.lon, N_BLOCKS, labels=False) * N_BLOCKS
                + pd.cut(agg.lat, N_BLOCKS, labels=False))


def spatial_split(seed):
    """Hold out whole blocks, never individual cells.
    A random split would put neighbouring cells on both sides
    and inflate the score."""
    blocks = agg.block.unique().copy()
    np.random.default_rng(seed).shuffle(blocks)
    held = set(blocks[:int(len(blocks) * TEST_FRAC)])
    return agg[~agg.block.isin(held)], agg[agg.block.isin(held)]


def new_model():
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                 n_jobs=-1, random_state=42)


# ---------------------------------------------------------------- evaluate
r2s, rmses, maes = [], [], []
for s in SEEDS:
    tr, te = spatial_split(s)
    m = new_model().fit(tr[FEATURES], tr.agbd)
    p = m.predict(te[FEATURES])
    r2s.append(r2_score(te.agbd, p))
    rmses.append(np.sqrt(mean_squared_error(te.agbd, p)))
    maes.append(mean_absolute_error(te.agbd, p))
    print(f"  seed {s:>4}: R2 {r2s[-1]:.3f}  RMSE {rmses[-1]:.2f}")

metrics = {
    "n_cells": int(len(agg)),
    "cell_size_m": CELL_M,
    "n_features": len(FEATURES),
    "r2_mean": round(float(np.mean(r2s)), 3),
    "r2_sd": round(float(np.std(r2s)), 3),
    "rmse_mean": round(float(np.mean(rmses)), 2),
    "rmse_sd": round(float(np.std(rmses)), 2),
    "mae_mean": round(float(np.mean(maes)), 2),
    "validation": "spatially-blocked, 5 repeats",
}
print("\n=== HEADLINE (use these in the deck) ===")
print(f"R2   {metrics['r2_mean']} +/- {metrics['r2_sd']}")
print(f"RMSE {metrics['rmse_mean']} +/- {metrics['rmse_sd']} Mg/ha")
json.dump(metrics, open("outputs/metrics.json", "w"), indent=2)

# ---------------------------------------------------------------- plots
tr, te = spatial_split(SEEDS[0])
model = new_model().fit(tr[FEATURES], tr.agbd)
pred = model.predict(te[FEATURES])

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(te.agbd, pred, s=8, alpha=0.35, edgecolors="none")
lim = [0, max(te.agbd.max(), pred.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=1, label="1:1 line")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("GEDI measured AGB (Mg/ha)")
ax.set_ylabel("Predicted AGB (Mg/ha)")
ax.set_title(f"Arboris — spatially-blocked validation\n"
             f"R² = {metrics['r2_mean']} ± {metrics['r2_sd']},  "
             f"RMSE = {metrics['rmse_mean']} Mg/ha")
ax.legend()
fig.tight_layout()
fig.savefig("outputs/scatter.png", dpi=160)

imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 7))
imp.plot.barh(ax=ax)
ax.set_xlabel("Random Forest feature importance")
ax.set_title("What the model actually uses")
fig.tight_layout()
fig.savefig("outputs/feature_importance.png", dpi=160)

# ---------------------------------------------------------------- final fit
final = new_model().fit(agg[FEATURES], agg.agbd)
joblib.dump({"model": final, "features": FEATURES,
             "cell_size_m": CELL_M, "metrics": metrics},
            "models/arboris_rf.pkl")
print("\nsaved models/arboris_rf.pkl, outputs/metrics.json, outputs/*.png")
