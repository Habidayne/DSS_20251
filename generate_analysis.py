"""
generate_analysis.py — Orchestrator: Sinh toàn bộ 14 biểu đồ cho Phần 2 & 3

Chạy:
    python generate_analysis.py

Output:
    outputs/part2/  -> 14 PNG files

Kiến trúc 2 model:
- OOS model  (train ≤2021, val 2022 OOS): cho Viz9 metrics + Viz11 SHAP trung thực
- Full model (train ≤2022)              : cho Viz10 forecast 2023-2024 (dùng toàn bộ data)
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Setup paths ──────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.utils import setup_logger, load_sales, SEED
from src.denoising import denoise_target
from src.prophet_model import fit_prophet, predict_prophet
from src.lgbm_model import fit_catboost_residual, predict_catboost_residual, make_future_safe_features, FEAT_COLS
from src.postprocess import blend_forecasts

np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────
CSV_DIR    = os.path.join(ROOT, "csv")
OUT_PART2  = os.path.join(ROOT, "outputs", "part2")
os.makedirs(OUT_PART2, exist_ok=True)

SALES_FILE      = os.path.join(CSV_DIR, "sales.csv")
INVENTORY_FILE  = os.path.join(CSV_DIR, "inventory.csv")
PRODUCTS_FILE   = os.path.join(CSV_DIR, "products.csv")
ORDERS_FILE     = os.path.join(CSV_DIR, "orders.csv")
ORDER_ITEMS_FILE= os.path.join(CSV_DIR, "order_items.csv")
PROMOTIONS_FILE = os.path.join(CSV_DIR, "promotions.csv")
WEB_TRAFFIC_FILE= os.path.join(CSV_DIR, "web_traffic.csv")

# OOS split: train ≤2021, val 2022 — metrics cho Viz9 + SHAP cho Viz11
OOS_TRAIN_END = "2021-12-31"
OOS_VAL_START = "2022-01-01"
OOS_VAL_END   = "2022-12-31"

# Full split: train ≤2022 — forecast 2023-2024 cho Viz10
FULL_TRAIN_END = "2022-12-31"
TEST_START     = "2023-01-01"
TEST_END       = "2024-07-01"

LOG_FILE = os.path.join(ROOT, "log_analysis.txt")
logger   = setup_logger(LOG_FILE)


def _load_exog(full_idx: pd.DatetimeIndex) -> dict:
    """Load exogenous regressors cho LightGBM (khớp FEAT_COLS: fill_rate / days_supply / overstock_pct / sessions)."""
    _web = pd.read_csv(WEB_TRAFFIC_FILE, parse_dates=["date"])
    sessions = _web.groupby("date")["sessions"].sum().reindex(full_idx)

    _inv = pd.read_csv(INVENTORY_FILE)
    _inv["snap"] = pd.to_datetime(_inv["year"].astype(str) + "-" + _inv["month"].astype(str) + "-01")
    def _invd(col):
        return _inv.groupby("snap")[col].mean().resample("D").ffill().reindex(full_idx)

    return {
        "sessions":       sessions,
        "overstock_pct":  _invd("overstock_flag"),   # mean of binary flag = overstock fraction
        "days_of_supply": _invd("days_of_supply"),
        "fill_rate":      _invd("fill_rate"),
    }


def _run_pipeline(
    df: pd.DataFrame,
    exog: dict,
    full_idx: pd.DatetimeIndex,
    train_end: str,
    val_start: str = None,
    val_end: str = None,
    compute_shap: bool = False,
    label: str = "",
) -> dict:
    """
    Chạy Prophet + LGBM pipeline cho một cặp (train_end, val split).

    Returns dict với keys:
        val_results, forecast_dfs, shap_data, all_forecast_errors
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    train_df    = df.loc[:train_end]
    val_results = {}
    forecast_dfs= {}
    shap_data   = {}
    all_errors  = {}
    _resid_stds = {}

    for target in ["Revenue", "COGS"]:
        clean_col    = f"Clean_{target}"
        train_series = train_df[clean_col].dropna()

        # ── Tầng 1: Prophet ──────────────────────────
        model        = fit_prophet(train_series, target_name=target)
        prophet_preds= predict_prophet(model, str(df.index.min().date()), TEST_END)

        # ── Residuals ────────────────────────────────
        train_prophet   = prophet_preds.loc[:train_end, "prophet_pred"]
        residual_train  = train_df[target].reindex(train_prophet.index) - train_prophet

        full_residuals  = pd.Series(np.nan, index=full_idx, name="residual")
        full_residuals.update(residual_train)
        full_trend      = prophet_preds["prophet_trend"].reindex(full_idx)
        X_full          = make_future_safe_features(full_residuals, full_trend, full_idx, exog)

        X_train  = X_full.loc[:train_end]
        y_train  = residual_train.reindex(X_train.index)
        lgbm_model = fit_catboost_residual(X_train, y_train, target_name=target)

        # ── OOS Validation metrics ───────────────────
        resid_std = df[target].std() * 0.15   # fallback nếu không có val
        if val_start and val_end:
            val_prophet      = prophet_preds.loc[val_start:val_end, "prophet_pred"]
            X_val            = X_full.loc[val_start:val_end]
            val_resid        = predict_catboost_residual(lgbm_model, X_val)
            val_final        = blend_forecasts(val_prophet, val_resid)
            actual_val       = df.loc[val_start:val_end, target].reindex(val_final.index).dropna()
            val_final_al     = val_final.reindex(actual_val.index)
            baseline         = df[target].shift(365).loc[val_start:val_end].reindex(actual_val.index)

            if target == "Revenue":
                val_results["Baseline (YoY Naive)"] = baseline
                val_results["Prophet Only"]          = val_prophet.reindex(actual_val.index)
                val_results["Gridbreaker"]           = val_final_al

            all_errors[target] = (actual_val - val_final_al) / val_final_al.replace(0, np.nan)
            resid_std          = (actual_val - val_final_al).std()
            _resid_stds[target]= resid_std

            mae  = mean_absolute_error(actual_val, val_final_al)
            rmse = np.sqrt(mean_squared_error(actual_val, val_final_al))
            r2   = r2_score(actual_val, val_final_al)
            tag  = f"OOS {val_start[:4]}" if label == "oos" else f"Val {val_start[:4]}"
            print(f"  {target} {tag} -> MAE={mae/1e6:.2f}M  RMSE={rmse/1e6:.2f}M  R²={r2:.4f}")

            # ── SHAP (OOS model trên OOS val data) ──
            if compute_shap and target == "Revenue":
                try:
                    import shap as _shap
                    explainer    = _shap.TreeExplainer(lgbm_model)
                    X_shap       = X_val.dropna().head(2000)
                    sv           = explainer.shap_values(X_shap)
                    shap_data["shap_values"]   = sv
                    shap_data["feature_names"] = FEAT_COLS
                    shap_data["X_sample"]      = X_shap
                    print(f"  SHAP computed: {X_shap.shape[0]} OOS samples, {len(FEAT_COLS)} features")
                except Exception as e:
                    print(f"  SHAP failed: {e}")
                    shap_data["shap_values"]   = np.zeros((100, len(FEAT_COLS)))
                    shap_data["feature_names"] = FEAT_COLS
                    shap_data["X_sample"]      = X_val.head(100)

        # ── Test forecast (2023-2024) ─────────────────
        test_prophet = prophet_preds.loc[TEST_START:TEST_END]
        X_test       = X_full.loc[TEST_START:TEST_END]
        test_resid   = predict_catboost_residual(lgbm_model, X_test)
        test_final   = blend_forecasts(test_prophet["prophet_pred"], test_resid)

        fc_df = pd.DataFrame({
            "forecast":  test_final,
            "lower_95":  (test_final - 1.96 * resid_std).clip(lower=0),
            "upper_95":  test_final + 1.96 * resid_std,
        }, index=test_final.index)
        forecast_dfs[target] = fc_df

    return {
        "val_results":   val_results,
        "forecast_dfs":  forecast_dfs,
        "shap_data":     shap_data,
        "all_errors":    all_errors,
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("  THE GRIDBREAKER — ANALYSIS PIPELINE")
    print("  Generating all 14 visualizations...")
    print("=" * 60)

    # ── 1. Load & Denoise ────────────────────────────────────
    print("\n[1/6] Loading data...")
    df = load_sales(SALES_FILE)
    df = denoise_target(df)
    print(f"  Sales: {df.index.min().date()} -> {df.index.max().date()} ({len(df)} rows)")

    full_idx = pd.date_range(df.index.min(), TEST_END)
    exog     = _load_exog(full_idx)

    # ── 2. OOS model (train ≤2021, val 2022): Viz9 metrics + SHAP ──
    print("\n[2/6] Running model pipeline for validation metrics...")
    print(f"  [OOS] train ≤{OOS_TRAIN_END}, val {OOS_VAL_START[:4]} (true out-of-sample)...")
    oos = _run_pipeline(df, exog, full_idx,
                        train_end=OOS_TRAIN_END,
                        val_start=OOS_VAL_START,
                        val_end=OOS_VAL_END,
                        compute_shap=True,
                        label="oos")
    val_results        = oos["val_results"]
    shap_data          = oos["shap_data"]
    all_forecast_errors= oos["all_errors"]

    # ── Full model (train ≤2022): Viz10 forecast ─────────────
    print(f"  [Full] train ≤{FULL_TRAIN_END} (best forecast for 2023-2024)...")
    full = _run_pipeline(df, exog, full_idx,
                         train_end=FULL_TRAIN_END,
                         val_start=OOS_VAL_START,  # dùng 2022 để ước std cho CI
                         val_end=OOS_VAL_END,
                         compute_shap=False,
                         label="full")
    forecast_dfs = full["forecast_dfs"]

    # ── 3. DESCRIPTIVE ──────────────────────────────────────
    print("\n[3/6] Generating Descriptive visualizations...")
    from src.analysis.descriptive import run_descriptive
    desc_paths = run_descriptive(
        sales=df, out_dir=OUT_PART2,
        order_items_path=ORDER_ITEMS_FILE,
        products_path=PRODUCTS_FILE,
        orders_path=ORDERS_FILE,
    )

    # ── 4. DIAGNOSTIC ───────────────────────────────────────
    print("\n[4/6] Generating Diagnostic visualizations...")
    from src.analysis.diagnostic import run_diagnostic
    diag_paths = run_diagnostic(
        sales=df, clean_revenue=df["Clean_Revenue"], out_dir=OUT_PART2,
        inventory_path=INVENTORY_FILE,
        promotions_path=PROMOTIONS_FILE,
        web_traffic_path=WEB_TRAFFIC_FILE,
    )

    # ── 5. PREDICTIVE ───────────────────────────────────────
    print("\n[5/6] Generating Predictive visualizations...")
    from src.analysis.predictive import run_predictive
    pred_paths = run_predictive(
        sales=df,
        val_results=val_results,
        forecast_df=forecast_dfs["Revenue"],
        shap_values=shap_data["shap_values"],
        feature_names=shap_data["feature_names"],
        X_sample=shap_data["X_sample"],
        out_dir=OUT_PART2,
    )

    # ── 6. PRESCRIPTIVE ─────────────────────────────────────
    print("\n[6/6] Generating Prescriptive visualizations...")
    from src.analysis.prescriptive import run_prescriptive
    presc_paths = run_prescriptive(
        sales=df,
        forecast_errors=all_forecast_errors["Revenue"],
        out_dir=OUT_PART2,
        inventory_path=INVENTORY_FILE,
        web_traffic_path=WEB_TRAFFIC_FILE,
    )

    # ── Summary ─────────────────────────────────────────────
    elapsed   = time.time() - t0
    all_paths = desc_paths + diag_paths + pred_paths + presc_paths
    print(f"\n{'='*60}")
    print(f"  DONE -- {len(all_paths)} charts generated in {elapsed:.1f}s")
    print(f"  Output: {OUT_PART2}")
    for i, p in enumerate(all_paths, 1):
        print(f"    Viz {i:2d}: {os.path.basename(p)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
