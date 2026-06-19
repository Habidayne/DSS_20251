"""
generate_analysis.py — Orchestrator: Sinh toàn bộ 14 biểu đồ cho Phần 2 & 3

Chạy:
    python generate_analysis.py

Output:
    outputs/part2/  -> 14 PNG files
    outputs/part3/  -> SHAP + model artifacts
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
from src.lgbm_model import fit_lgbm_residual, predict_lgbm_residual, make_future_safe_features, FEAT_COLS
from src.postprocess import blend_forecasts

np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────
CSV_DIR    = os.path.join(ROOT, "csv")
OUT_PART2  = os.path.join(ROOT, "outputs", "part2")
OUT_PART3  = os.path.join(ROOT, "outputs", "part3")
os.makedirs(OUT_PART2, exist_ok=True)
os.makedirs(OUT_PART3, exist_ok=True)

SALES_FILE      = os.path.join(CSV_DIR, "sales.csv")
INVENTORY_FILE  = os.path.join(CSV_DIR, "inventory.csv")
PRODUCTS_FILE   = os.path.join(CSV_DIR, "products.csv")
ORDERS_FILE     = os.path.join(CSV_DIR, "orders.csv")
ORDER_ITEMS_FILE= os.path.join(CSV_DIR, "order_items.csv")
PROMOTIONS_FILE = os.path.join(CSV_DIR, "promotions.csv")
WEB_TRAFFIC_FILE= os.path.join(CSV_DIR, "web_traffic.csv")

TRAIN_END  = "2022-12-31"
VAL_START  = "2021-01-01"
VAL_END    = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END   = "2024-07-01"

LOG_FILE = os.path.join(ROOT, "log_analysis.txt")
logger   = setup_logger(LOG_FILE)


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
    train = df.loc[:TRAIN_END]
    print(f"  Sales: {df.index.min().date()} -> {df.index.max().date()} ({len(df)} rows)")

    # ── 2. Run model pipeline (for Predictive/Prescriptive viz) ──
    print("\n[2/6] Running model pipeline for validation metrics...")
    full_idx = pd.date_range(df.index.min(), TEST_END)

    # Exogenous regressors (khớp FEAT_COLS mới — fill_rate/days_supply/overstock/sessions lag365)
    _web = pd.read_csv(WEB_TRAFFIC_FILE, parse_dates=["date"])
    _sessions = _web.groupby("date")["sessions"].sum().reindex(full_idx)
    _inv = pd.read_csv(INVENTORY_FILE)
    _inv["snap"] = pd.to_datetime(_inv["year"].astype(str) + "-" + _inv["month"].astype(str) + "-01")
    def _invd(c):
        return _inv.groupby("snap")[c].mean().resample("D").ffill().reindex(full_idx)
    exog = {"sessions": _sessions, "overstock_pct": _invd("overstock_flag"),
            "days_of_supply": _invd("days_of_supply"), "fill_rate": _invd("fill_rate")}

    val_results = {}
    forecast_dfs = {}
    shap_data = {}
    all_forecast_errors = {}

    for target in ["Revenue", "COGS"]:
        clean_col = f"Clean_{target}"
        train_series = train[clean_col].dropna()

        # Prophet
        model = fit_prophet(train_series, target_name=target)
        prophet_preds = predict_prophet(model, str(df.index.min().date()), TEST_END)

        # Residual
        train_prophet = prophet_preds.loc[:TRAIN_END, "prophet_pred"]
        residual_train = train[target].reindex(train_prophet.index) - train_prophet

        # LGBM features
        full_residuals = pd.Series(np.nan, index=full_idx, name="residual")
        full_residuals.update(residual_train)
        full_trend = prophet_preds["prophet_trend"].reindex(full_idx)
        X_full = make_future_safe_features(full_residuals, full_trend, full_idx, exog)

        X_train = X_full.loc[:TRAIN_END]
        y_train = residual_train.reindex(X_train.index)
        lgbm_model = fit_lgbm_residual(X_train, y_train, target_name=target)

        # Validation predictions
        val_prophet = prophet_preds.loc[VAL_START:VAL_END, "prophet_pred"]
        X_val = X_full.loc[VAL_START:VAL_END]
        val_resid = predict_lgbm_residual(lgbm_model, X_val)
        val_final = blend_forecasts(val_prophet, val_resid)

        actual_val = df.loc[VAL_START:VAL_END, target].reindex(val_final.index).dropna()
        val_final_aligned = val_final.reindex(actual_val.index)

        # Baseline: YoY seasonal naive
        baseline = df[target].shift(365).loc[VAL_START:VAL_END].reindex(actual_val.index)

        if target == "Revenue":
            val_results["Baseline (YoY Naive)"] = baseline
            val_results["Prophet Only"]          = val_prophet.reindex(actual_val.index)
            val_results["Gridbreaker"]           = val_final_aligned

        # Forecast errors for prescriptive analysis (actual - forecast) / forecast
        forecast_error = (actual_val - val_final_aligned) / val_final_aligned
        all_forecast_errors[target] = forecast_error

        # Test period forecast with prediction intervals
        test_prophet_full = prophet_preds.loc[TEST_START:TEST_END]
        X_test = X_full.loc[TEST_START:TEST_END]
        lgbm_resid_pred = predict_lgbm_residual(lgbm_model, X_test)
        test_final = blend_forecasts(test_prophet_full["prophet_pred"], lgbm_resid_pred)

        # Prediction intervals from Prophet + residual uncertainty
        prophet_lower = test_prophet_full.get("prophet_lower", test_prophet_full["prophet_pred"] * 0.85)
        prophet_upper = test_prophet_full.get("prophet_upper", test_prophet_full["prophet_pred"] * 1.15)

        # Use validation error std to estimate residual uncertainty
        resid_std = (actual_val - val_final_aligned).std()

        fc_df = pd.DataFrame({
            "forecast":  test_final,
            "lower_95":  test_final - 1.96 * resid_std,
            "upper_95":  test_final + 1.96 * resid_std,
        }, index=test_final.index)
        fc_df["lower_95"] = fc_df["lower_95"].clip(lower=0)
        forecast_dfs[target] = fc_df

        # SHAP
        if target == "Revenue":
            try:
                import shap
                explainer = shap.TreeExplainer(lgbm_model)
                X_shap = X_val.dropna().head(2000)
                sv = explainer.shap_values(X_shap)
                shap_data["shap_values"]    = sv
                shap_data["feature_names"]  = FEAT_COLS
                shap_data["X_sample"]       = X_shap
                print(f"  SHAP computed: {X_shap.shape[0]} samples, {len(FEAT_COLS)} features")
            except Exception as e:
                print(f"  SHAP failed: {e}")
                shap_data["shap_values"]   = np.zeros((100, len(FEAT_COLS)))
                shap_data["feature_names"] = FEAT_COLS
                shap_data["X_sample"]      = X_val.head(100)

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae  = mean_absolute_error(actual_val, val_final_aligned)
        rmse = np.sqrt(mean_squared_error(actual_val, val_final_aligned))
        r2   = r2_score(actual_val, val_final_aligned)
        print(f"  {target} Val -> MAE={mae/1e6:.2f}M  RMSE={rmse/1e6:.2f}M  R²={r2:.4f}")

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
    elapsed = time.time() - t0
    all_paths = desc_paths + diag_paths + pred_paths + presc_paths
    print(f"\n{'='*60}")
    print(f"  DONE -- {len(all_paths)} charts generated in {elapsed:.1f}s")
    print(f"  Output: {OUT_PART2}")
    for i, p in enumerate(all_paths, 1):
        print(f"    Viz {i:2d}: {os.path.basename(p)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
