"""
Arboris - validation diagnostics beyond the headline R2.
Run:  python src/diagnostics.py

Answers two questions a judge will actually ask:

  1. "Where does your model break down?"
     -> residuals vs predicted, which exposes the high-biomass saturation that
        every GEDI-trained AGB model has. We show it rather than hide it.

  2. "You give me one number per cell. How do I know which cells to trust?"
     -> a Random Forest's 500 trees disagree more where the model is unsure.
        We measure that spread AND check, on held-out data, whether it actually
        predicts error. An uncertainty layer nobody validated is decoration.

Input :  data/arboris_training.csv, models/arboris_rf.pkl
Output:  outputs/residuals.png
         outputs/uncertainty_calibration.png
         outputs/diagnostics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Must mirror src/data/train_model.py exactly, or the hold-out is not the hold-out.
CSV         = Path("data/arboris_training.csv")
MODEL       = Path("models/arboris_rf.pkl")
CELL_M      = 200
MIN_SHOTS   = 3
SE_MAX_FRAC = 0.5
N_BLOCKS    = 10
TEST_FRAC   = 0.25
SEED        = 42
DEG_PER_CELL = CELL_M / 111_320.0

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- rebuild the split
df = pd.read_csv(CSV)
coords = df[".geo"].map(lambda s: json.loads(s)["coordinates"])
df["lon"] = [c[0] for c in coords]
df["lat"] = [c[1] for c in coords]
df = df[(df.agbd_se / df.agbd) <= SE_MAX_FRAC].copy()

FEATURES = joblib.load(MODEL)["features"]

df["cx"] = np.floor(df.lon / DEG_PER_CELL).astype(int)
df["cy"] = np.floor(df.lat / DEG_PER_CELL).astype(int)
agg = (df.groupby(["cx", "cy"])
         .agg(**{f: (f, "mean") for f in FEATURES},
              agbd=("agbd", "mean"), lon=("lon", "mean"), lat=("lat", "mean"),
              nshot=("agbd", "size"))
         .reset_index())
agg = agg[agg.nshot >= MIN_SHOTS].reset_index(drop=True)
agg["block"] = (pd.cut(agg.lon, N_BLOCKS, labels=False) * N_BLOCKS
                + pd.cut(agg.lat, N_BLOCKS, labels=False))

blocks = agg.block.unique().copy()
np.random.default_rng(SEED).shuffle(blocks)
held = set(blocks[:int(len(blocks) * TEST_FRAC)])
tr, te = agg[~agg.block.isin(held)], agg[agg.block.isin(held)]

# Leakage assertion (I2 / §6.6): no block may appear on both sides.
overlap = set(tr.block) & set(te.block)
if overlap:
    raise AssertionError(f"block leakage across the split: {sorted(overlap)[:5]}")
print(f"train {len(tr):,} cells / test {len(te):,} cells over "
      f"{len(held)} held-out blocks (no block overlap)")

model = RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                              n_jobs=-1, random_state=SEED).fit(tr[FEATURES], tr.agbd)

Xte = te[FEATURES].to_numpy()
pred = model.predict(Xte)
obs = te.agbd.to_numpy()
resid = pred - obs

# ---------------------------------------------------------------- tree disagreement
# The spread of the 500 individual tree predictions. This is MODEL DISAGREEMENT,
# not a calibrated confidence interval - and it is labelled that way everywhere.
per_tree = np.stack([t.predict(Xte) for t in model.estimators_])
spread = per_tree.std(axis=0)

abs_err = np.abs(resid)
rho = float(np.corrcoef(spread, abs_err)[0, 1])

# Does the flag actually work? Split the hold-out into thirds by spread.
q = np.quantile(spread, [1/3, 2/3])
band = np.digitize(spread, q)
names = ["low (most confident)", "medium", "high (least confident)"]
bands = []
for i, nm in enumerate(names):
    m = band == i
    bands.append({"band": nm, "n": int(m.sum()),
                  "mean_spread_mg_ha": round(float(spread[m].mean()), 2),
                  "rmse_mg_ha": round(float(np.sqrt((resid[m] ** 2).mean())), 2),
                  "mae_mg_ha": round(float(abs_err[m].mean()), 2)})
    print(f"  {nm:24s} n={m.sum():4d}  spread {spread[m].mean():5.1f}  "
          f"RMSE {np.sqrt((resid[m]**2).mean()):5.1f} Mg/ha")

ratio = bands[2]["rmse_mg_ha"] / bands[0]["rmse_mg_ha"]
print(f"\ncorr(tree spread, |error|) = {rho:.3f}")
print(f"least-confident third has {ratio:.2f}x the RMSE of the most-confident third")

# ---------------------------------------------------------------- saturation
# Where does the model stop tracking reality? Bin observed AGB and look at bias.
edges = np.array([0, 50, 100, 150, 200, 250, 300, 1e9])
sat = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (obs >= lo) & (obs < hi)
    if m.sum() < 10:
        continue
    sat.append({"obs_range_mg_ha": f"{int(lo)}-{'+' if hi > 1e8 else int(hi)}",
                "n": int(m.sum()),
                "mean_observed_mg_ha": round(float(obs[m].mean()), 1),
                "mean_predicted_mg_ha": round(float(pred[m].mean()), 1),
                "bias_mg_ha": round(float(resid[m].mean()), 1)})

# ---------------------------------------------------------------- plots
plt.rcParams.update({"font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(7.2, 5))
ax.axhline(0, color="#444", lw=1.2, zorder=1)
ax.scatter(pred, resid, s=9, alpha=0.3, color="#2e7d4f", edgecolors="none", zorder=2)
o = np.argsort(pred)
w = max(25, len(pred) // 20)
roll = pd.Series(resid[o]).rolling(w, center=True, min_periods=w // 3).mean()
ax.plot(pred[o], roll, color="#c9721a", lw=2, zorder=3, label=f"rolling mean (window {w})")
ax.set_xlabel("Predicted AGB (Mg/ha)")
ax.set_ylabel("Residual  (predicted - GEDI observed, Mg/ha)")
ax.set_title("Residuals vs predicted - spatially-blocked hold-out\n"
             "Downward slope at high biomass is saturation: a known, expected "
             "limit of optical/SAR\nfeatures in dense canopy, not a bug.",
             fontsize=10.5, loc="left")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "residuals.png", dpi=160)
print(f"\nwrote {OUT/'residuals.png'}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
a1.scatter(spread, abs_err, s=9, alpha=0.3, color="#2e7d4f", edgecolors="none")
a1.set_xlabel("Tree disagreement, 1 s.d. across 500 trees (Mg/ha)")
a1.set_ylabel("|error| vs GEDI (Mg/ha)")
a1.set_title(f"Does disagreement predict error?   r = {rho:.2f}", loc="left")

a2.bar([b["band"] for b in bands], [b["rmse_mg_ha"] for b in bands],
       color=["#2e7d4f", "#8aa832", "#c9721a"])
for i, b in enumerate(bands):
    a2.text(i, b["rmse_mg_ha"] + 1, f'{b["rmse_mg_ha"]:.0f}', ha="center", fontsize=10)
a2.set_ylabel("RMSE on hold-out (Mg/ha)")
a2.set_title(f"Confidence band vs actual error   ({ratio:.2f}x spread)", loc="left")
a2.tick_params(axis="x", labelrotation=12)
fig.tight_layout(); fig.savefig(OUT / "uncertainty_calibration.png", dpi=160)
print(f"wrote {OUT/'uncertainty_calibration.png'}")

# ---------------------------------------------------------------- json
diag = {
    "split": {"strategy": "spatial_block", "n_blocks_grid": N_BLOCKS,
              "seed": SEED, "n_train": int(len(tr)), "n_test": int(len(te)),
              "n_held_out_blocks": int(len(held)), "block_leakage": False},
    "holdout": {"r2": round(float(1 - (resid ** 2).sum()
                                  / ((obs - obs.mean()) ** 2).sum()), 3),
                "rmse_mg_ha": round(float(np.sqrt((resid ** 2).mean())), 2),
                "mae_mg_ha": round(float(abs_err.mean()), 2),
                "bias_mg_ha": round(float(resid.mean()), 2)},
    "uncertainty": {
        "definition": "1 s.d. of the 500 individual tree predictions; a measure of "
                      "MODEL DISAGREEMENT, not a calibrated confidence interval",
        "corr_spread_vs_abs_error": round(rho, 3),
        "rmse_ratio_high_over_low_band": round(float(ratio), 2),
        "bands": bands},
    "saturation": sat,
}
(OUT / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
print(f"wrote {OUT/'diagnostics.json'}")
