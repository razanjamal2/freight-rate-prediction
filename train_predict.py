"""Freight rate prediction pipeline.

Steps: load -> clean -> features -> rolling-origin validation -> final fit ->
write validation_predictions.csv and the completed December chart inputs.

Usage:
    python train_predict.py --data-dir data --output-dir outputs
    python train_predict.py --skip-cv        # skip validation, only fit + predict
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T0 = pd.Timestamp("2025-01-01")
MAX_WEIGHT = 47_500.0  # the data is capped here
OUTLIER_SIGMAS = 5.0   # robust-sigma cut for label cleaning
EQUIP = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
SEEDS = (0, 1, 2)


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def date_level_signals(*frames: pd.DataFrame) -> pd.DataFrame:
    """Per-date market_index (median) and quote_signal (mean) from FEATURES only.

    No labels are used. Row-level market_index is noisy and sometimes missing, while
    the daily aggregate is stable, so it also fills gaps and supplies values for the
    December chart rows, which have no market_index / quote_signal columns.
    """
    cols = ["date", "market_index", "quote_signal"]
    allf = pd.concat([f[cols] for f in frames])
    return (allf.groupby("date")
            .agg(d_mi=("market_index", "median"), d_qs=("quote_signal", "mean"))
            .reset_index())


def build_features(x: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    x = x.merge(signals, on="date", how="left")
    f = pd.DataFrame(index=x.index)
    f["ld"] = np.log(x["distance"])
    f["dist"] = x["distance"]
    f["eq"] = x["equipment"].map(EQUIP)
    # data-quality fixes: negative weights are sign errors -> abs; cap at 47,500;
    # missing weight stays NaN (LightGBM handles it; the linear model uses the median).
    f["w"] = x["weight"].abs().clip(upper=MAX_WEIGHT)
    f["d_mi"], f["d_qs"] = x["d_mi"], x["d_qs"]
    f["mi"] = x["market_index"].fillna(x["d_mi"])  # missing market_index -> daily value
    f["qs"] = x["quote_signal"]
    f["dow"] = x["date"].dt.dayofweek
    # coordinates generalise to cities never seen in training (city names do not)
    for c in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        f[c] = x[c]
    f["t"] = (x["date"] - T0).dt.days.values
    return f


BASE_COLS = ["ld", "dist", "eq", "w", "d_mi", "d_qs", "mi", "qs", "dow",
             "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


def linear_design(f: pd.DataFrame) -> np.ndarray:
    z = pd.DataFrame({
        "c": 1.0, "ld": f["ld"], "ld2": f["ld"] ** 2, "mi": f["mi"], "qs": f["qs"],
        "w": f["w"].fillna(31_000.0) / 1e4,
        "flat": (f["eq"] == 1) * 1.0, "reef": (f["eq"] == 2) * 1.0,
        "t": f["t"] / 100.0,
    })
    return z.values


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def make_gbm(seed: int, n: int = 500) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=n, learning_rate=0.05, num_leaves=31, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        objective="huber", alpha=0.3, random_state=seed, verbose=-1)


def inlier_mask(f: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Flag rows whose log-rate is wildly off a first-pass fit (corrupt labels)."""
    m = make_gbm(0, 300).fit(f[BASE_COLS], y)
    r = y - m.predict(f[BASE_COLS])
    med = np.median(r)
    sigma = 1.4826 * np.median(np.abs(r - med))
    return np.abs(r - med) < OUTLIER_SIGMAS * sigma


def fit_predict(f_tr: pd.DataFrame, y_tr: np.ndarray, f_te: pd.DataFrame,
                clean: bool = True) -> np.ndarray:
    """Blend of (A) a GBM that sees the time index and (B) linear trend + GBM on the residual.

    Both extrapolate the slow market drift differently (A holds the last level, B extends a
    linear trend); averaging hedges between them. Returns log-rate predictions.
    """
    if clean:
        ok = inlier_mask(f_tr, y_tr)
        f_tr, y_tr = f_tr[ok], y_tr[ok]
    # A: tree with time index
    pa = np.mean([make_gbm(s).fit(f_tr[BASE_COLS + ["t"]], y_tr).predict(f_te[BASE_COLS + ["t"]])
                  for s in SEEDS], axis=0)
    # B: linear (log-distance, signals, equipment, weight, linear trend) + GBM on residual
    ztr, zte = linear_design(f_tr), linear_design(f_te)
    beta = np.linalg.lstsq(ztr, y_tr, rcond=None)[0]
    resid = y_tr - ztr @ beta
    pb = zte @ beta + np.mean([make_gbm(s).fit(f_tr[BASE_COLS], resid).predict(f_te[BASE_COLS])
                               for s in SEEDS], axis=0)
    return (pa + pb) / 2.0


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    a, b = np.exp(y_true_log), np.exp(y_pred_log)
    return {"MAE": float(np.mean(np.abs(a - b))),
            "RMSE": float(np.sqrt(np.mean((a - b) ** 2))),
            "MAPE_%": float(np.mean(np.abs(a - b) / a) * 100),
            "MdAPE_%": float(np.median(np.abs(a - b) / a) * 100),
            "bias_%": float(np.mean(np.log(b / a)) * 100)}


def run_validation(f: pd.DataFrame, y: np.ndarray, d: pd.DataFrame) -> list[dict]:
    """Forward-chaining 2-month folds (train strictly before, test strictly after) plus a
    final fold that also hides 8 cities, mimicking validation.csv (later dates + unseen cities)."""
    rows = []
    folds = [("2025-03-01", "2025-05-01"), ("2025-05-01", "2025-07-01"),
             ("2025-07-01", "2025-09-01"), ("2025-09-01", "2025-11-01")]
    rng = np.random.RandomState(7)
    hold = set(rng.choice(sorted(set(d.pickup) | set(d.delivery)), 8, replace=False))
    has_hold = (d.pickup.isin(hold) | d.delivery.isin(hold)).values
    specs = [(f"train<{lo} -> test {lo}..{hi}", d.date < lo, (d.date >= lo) & (d.date < hi))
             for lo, hi in folds]
    specs.append(("train<2025-09-01 w/o 8 cities -> Sep-Oct on those cities (unseen)",
                  (d.date < "2025-09-01") & ~has_hold,
                  (d.date >= "2025-09-01") & has_hold))
    for name, trm, tem in specs:
        trm, tem = np.asarray(trm), np.asarray(tem)
        p = fit_predict(f[trm], y[trm], f[tem])
        yt = y[tem]
        clean = inlier_mask(f[tem], yt)  # labels that are not corrupt
        rows.append({"fold": name, "n_test": int(tem.sum()),
                     "all_rows": metrics(yt, p), "clean_rows": metrics(yt[clean], p[clean]),
                     "corrupt_label_share_%": float((~clean).mean() * 100)})
        print(f"{name}: clean MAPE={rows[-1]['clean_rows']['MAPE_%']:.2f}% "
              f"bias={rows[-1]['clean_rows']['bias_%']:+.2f}%")
    return rows


# ----------------------------------------------------------------------------
# December chart inputs
# ----------------------------------------------------------------------------
def december_features(dec: pd.DataFrame, coords: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    x = dec.copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x.merge(coords.add_prefix("pickup_"), left_on="pickup", right_index=True, how="left")
    x = x.merge(coords.add_prefix("delivery_"), left_on="delivery", right_index=True, how="left")
    # no row-level signals are supplied -> use the day's aggregate from validation.csv
    sig = signals.set_index("date")
    x["market_index"] = x["date"].map(sig["d_mi"])
    x["quote_signal"] = x["date"].map(sig["d_qs"])
    return x


def city_coords(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for fr in frames:
        for side in ("pickup", "delivery"):
            parts.append(fr[[side, f"{side}_lat", f"{side}_lon"]]
                         .set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


# ----------------------------------------------------------------------------
def save_figures(d: pd.DataFrame, y: np.ndarray, f: pd.DataFrame, out: Path) -> None:
    ok = inlier_mask(f, y)
    rpm = np.log(d.posted_rate / d.distance)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)
    ax[0].hist(np.log(d.posted_rate) - np.log(d.groupby(["equipment"]).posted_rate.transform("median")),
               bins=80, color="#064A56")
    ax[0].set_title("Log rate vs equipment median (pre-clean)")
    m = d.assign(rpm=d.posted_rate / d.distance, ok=ok)
    mm = m[m.ok].groupby(m.date.dt.to_period("M")).rpm.median()
    ax[1].plot(mm.index.astype(str), mm.values, marker="o", color="#064A56")
    ax[1].set_title("Median $/mile by month (clean rows)")
    ax[1].tick_params(axis="x", rotation=45)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "eda_overview.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--skip-cv", action="store_true")
    a = ap.parse_args()
    data, out = Path(a.data_dir), Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(data / "train_test.csv", parse_dates=["date"])
    v = pd.read_csv(data / "validation.csv", parse_dates=["date"])
    tmpl = pd.read_csv(data / "validation_predictions_template.csv")
    dec = pd.read_csv(data / "december_chart_inputs.csv")

    signals = date_level_signals(d, v)
    f_d, f_v = build_features(d, signals), build_features(v, signals)
    y = np.log(d["posted_rate"].values)

    if not a.skip_cv:
        results = run_validation(f_d, y, d)
        (out / "validation_metrics.json").write_text(json.dumps(results, indent=2))
    save_figures(d, y, f_d, out)

    # ---- final fit on ALL labelled data, predict validation.csv + December ----
    coords = city_coords(d, v)
    dec_x = december_features(dec, coords, signals)
    f_dec = build_features(dec_x, signals)
    preds = np.exp(fit_predict(f_d, y, pd.concat([f_v, f_dec], ignore_index=True)))
    p_val, p_dec = preds[: len(f_v)], preds[len(f_v):]

    sub = tmpl[["load_id"]].merge(
        pd.DataFrame({"load_id": v["load_id"], "predicted_rate": np.round(p_val, 2)}),
        on="load_id", how="left")
    assert sub["predicted_rate"].notna().all() and len(sub) == 12_000
    sub.to_csv(out / "validation_predictions.csv", index=False)

    dec_out = dec.copy()
    dec_out["predicted_rate"] = np.round(p_dec, 2)
    dec_out.to_csv(out / "december_chart_inputs.csv", index=False)
    print(f"Wrote {out/'validation_predictions.csv'} and {out/'december_chart_inputs.csv'}")


if __name__ == "__main__":
    main()
