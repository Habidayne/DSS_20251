"""
pipeline.py — The Gridbreaker — Main Orchestrator

Pipeline du doan Revenue & COGS hang ngay cho ky test 01/01/2023 -> 01/07/2024.
4 buoc:
  1. Target Denoising   (src/denoising)
  2. Prophet             (src/prophet_model)
  3. LightGBM Residuals  (src/lgbm_model)
  4. Blend & Postprocess (src/postprocess)

Usage:
    python pipeline.py
"""
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -- Setup paths --
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.utils import setup_logger, mape, load_sales, SEED
from src.denoising import denoise_target
from src.prophet_model import fit_prophet, predict_prophet
from src.lgbm_model import fit_lgbm_residual, predict_lgbm_residual, make_future_safe_features
from src.postprocess import blend_forecasts, postprocess

np.random.seed(SEED)

# -- Config --
CSV_DIR         = os.path.join(ROOT, "csv")
SALES_FILE      = os.path.join(CSV_DIR, "sales.csv")
INVENTORY_FILE  = os.path.join(CSV_DIR, "inventory.csv")
WEB_FILE        = os.path.join(CSV_DIR, "web_traffic.csv")
SAMPLE_SUB_FILE = os.path.join(CSV_DIR, "sample_submission.csv")
OUTPUT_FILE     = os.path.join(ROOT, "submission.csv")
LOG_FILE        = os.path.join(ROOT, "log.txt")

TRAIN_END  = "2021-12-31"
VAL_START  = "2022-01-01"
VAL_END    = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END   = "2024-07-01"

logger = setup_logger(LOG_FILE)
logger.info("THE GRIDBREAKER PIPELINE - START")
logger.info(f"Train: ... -> {TRAIN_END}  |  Val: {VAL_START} -> {VAL_END}  |  Test: {TEST_START} -> {TEST_END}")


# ============================================================
# HELPER: Optuna hyperparameter search
# ============================================================

def _optuna_tune(X_train_clean: pd.DataFrame, y_train_clean: pd.Series) -> dict:
    """TimeSeriesSplit cross-validation + Optuna on MAE. Returns best_params dict."""
    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from src.lgbm_model.train_lgbm import EARLY_STOPPING_ROUNDS

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("  Running Optuna (50 trials, TimeSeriesSplit n=3)...")

    def objective(trial):
        params = {
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 7),
            "num_leaves":        trial.suggest_int("num_leaves", 8, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_estimators": 1000,
            "random_state": SEED,
            "verbose": -1,
        }
        tscv = TimeSeriesSplit(n_splits=3)
        mae_scores = []
        for tr_idx, vl_idx in tscv.split(X_train_clean):
            X_tr = X_train_clean.iloc[tr_idx]
            y_tr = y_train_clean.iloc[tr_idx]
            X_vl = X_train_clean.iloc[vl_idx]
            y_vl = y_train_clean.iloc[vl_idx]
            m = lgb.LGBMRegressor(**params)
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_vl, y_vl)],
                eval_metric="mae",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            mae_scores.append(mean_absolute_error(y_vl, m.predict(X_vl)))
        return np.mean(mae_scores)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    best = study.best_params
    best.update({"random_state": SEED, "verbose": -1, "n_estimators": 1000})
    logger.info(f"  Optuna best: {best}")
    return best


# ============================================================
# HELPER: Full Prophet + LGBM pipeline for one target
# ============================================================

def _forecast_one_target(target: str, df: pd.DataFrame, full_idx: pd.DatetimeIndex,
                         extra_regressors: dict = None):
    """
    Run Prophet -> LightGBM-residual -> Recursive Rolling Forecast for a single target.

    Args:
        target:   "Revenue" (COGS được suy ra riêng qua _seasonal_ratio)
        df:       denoised DataFrame (index=Date, columns include target & Clean_target)
        full_idx: date range from first train date to TEST_END
        extra_regressors: dict {name: daily Series} biến ngoại sinh (lag365 trong LGBM)

    Returns:
        (test_pred, val_pred):
          test_pred — pd.Series daily predictions TEST_START -> TEST_END
          val_pred  — pd.Series predictions trên 2022 validation (để ghép COGS metric)
    """
    from src.lgbm_model.train_lgbm import FEAT_COLS

    logger.info("=" * 50)
    logger.info(f"TARGET: {target}")
    logger.info("=" * 50)

    scale, unit = 1e6, "M VND"   # chỉ Revenue đi qua hàm này (COGS dùng _seasonal_ratio)

    train      = df.loc[:TRAIN_END]
    clean_col  = f"Clean_{target}"
    train_series = train[clean_col].dropna()

    # ── Prophet ──────────────────────────────────────────────
    logger.info(f"  [Prophet] Fitting {target}...")
    model_p      = fit_prophet(train_series, target_name=target)
    prophet_preds = predict_prophet(model_p, str(df.index.min().date()), TEST_END)
    train_prophet = prophet_preds.loc[:TRAIN_END, "prophet_pred"]
    residual_train = train[target].reindex(train_prophet.index) - train_prophet

    # Full residual series (NaN in val/test — to be filled progressively)
    full_residuals = pd.Series(np.nan, index=full_idx, name="residual")
    full_residuals.update(residual_train)
    full_trend = prophet_preds["prophet_trend"].reindex(full_idx)

    X_full = make_future_safe_features(full_residuals, full_trend, full_idx, extra_regressors)

    # ── Prepare train / val splits ────────────────────────────
    X_train   = X_full.loc[:TRAIN_END]
    y_train   = residual_train.reindex(X_train.index)

    val_prophet = prophet_preds.loc[VAL_START:VAL_END, "prophet_pred"]
    X_val       = X_full.loc[VAL_START:VAL_END]
    y_val       = (
        df.loc[VAL_START:VAL_END, target].reindex(X_val.index)
        - val_prophet.reindex(X_val.index)
    )

    # Drop rows with NaN (lag_365 needs >= 1 year of history)
    X_train_clean = X_train[FEAT_COLS].dropna()
    y_train_clean = y_train.reindex(X_train_clean.index).dropna()
    X_train_clean = X_train_clean.reindex(y_train_clean.index)

    X_val_clean = X_val[FEAT_COLS].dropna()
    y_val_clean = y_val.reindex(X_val_clean.index).dropna()
    X_val_clean = X_val_clean.reindex(y_val_clean.index)

    # ── Optuna tuning ─────────────────────────────────────────
    best_params = _optuna_tune(X_train_clean, y_train_clean)

    # ── LightGBM fit ──────────────────────────────────────────
    lgbm_model = fit_lgbm_residual(
        X_train_clean, y_train_clean,
        target_name=target,
        params=best_params,
        eval_set=(X_val_clean, y_val_clean),
    )

    # ── Out-of-sample validation metrics ─────────────────────
    val_resid   = predict_lgbm_residual(lgbm_model, X_val)
    val_final   = blend_forecasts(val_prophet, val_resid)
    actual_val  = df.loc[VAL_START:VAL_END, target].reindex(val_final.index).dropna()
    val_aligned = val_final.reindex(actual_val.index)

    mae_v  = mean_absolute_error(actual_val, val_aligned)
    rmse_v = np.sqrt(mean_squared_error(actual_val, val_aligned))
    r2_v   = r2_score(actual_val, val_aligned)
    mape_v = mape(actual_val.values, val_aligned.values)
    logger.info(f"  [{target}] Validation {VAL_START[:4]}:")
    logger.info(f"     MAE  = {mae_v/scale:.4f}{unit}")
    logger.info(f"     RMSE = {rmse_v/scale:.4f}{unit}")
    logger.info(f"     R2   = {r2_v:.4f}")
    logger.info(f"     MAPE = {mape_v:.2f}%")

    # Patch 2022 actual residuals so lag_365 for 2023 is grounded in reality
    residual_val_actual = actual_val - val_prophet.reindex(actual_val.index)
    full_residuals.update(residual_val_actual)

    # ── Recursive Rolling Forecast ────────────────────────────
    logger.info(f"  [{target}] Rolling forecast {TEST_START} -> {TEST_END}...")

    # Compounding error baseline: magnitude of actual residuals in 2022
    baseline_resid_mae = residual_val_actual.abs().mean()
    logger.info(f"  [{target}] Baseline residual MAE (2022 actual): {baseline_resid_mae/scale:.4f}{unit}")

    residuals_rolling = full_residuals.copy()
    test_years        = range(pd.Timestamp(TEST_START).year, pd.Timestamp(TEST_END).year + 1)
    all_predictions   = []

    for year in test_years:
        year_start = f"{year}-01-01"
        year_end   = f"{year}-12-31" if year < pd.Timestamp(TEST_END).year else TEST_END

        # Rebuild features from updated residuals (includes proxy from previous year)
        X_full_iter = make_future_safe_features(residuals_rolling, full_trend, full_idx, extra_regressors)
        nan_count   = X_full_iter.loc[year_start:year_end][FEAT_COLS].isna().sum().sum()
        logger.info(f"  [{target}][{year}] NaN in X_test: {nan_count}")

        test_prophet_yr = prophet_preds.loc[year_start:year_end, "prophet_pred"]
        X_test_yr       = X_full_iter.loc[year_start:year_end]
        test_resid_yr   = predict_lgbm_residual(lgbm_model, X_test_yr)
        test_pred_yr    = blend_forecasts(test_prophet_yr, test_resid_yr)
        all_predictions.append(test_pred_yr)

        # Compounding error log: proxy residual magnitude vs 2022 baseline
        proxy_resid     = test_pred_yr - test_prophet_yr
        proxy_resid_mae = proxy_resid.abs().mean()
        drift_ratio     = proxy_resid_mae / baseline_resid_mae if baseline_resid_mae > 0 else float("nan")
        logger.info(
            f"  [{target}][{year}] Proxy residual MAE: {proxy_resid_mae/scale:.4f}{unit} "
            f"(drift x{drift_ratio:.2f} vs 2022 baseline)"
        )

        # Feed proxy into rolling residual buffer for next year's lag_365 features
        residuals_rolling.update(proxy_resid)

    return pd.concat(all_predictions), val_aligned


# ============================================================
# HELPER: Seasonal ratio forecast (COGS/Revenue)
# ============================================================

def _seasonal_ratio(ratio_hist: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """
    Dự báo ratio = COGS/Revenue bằng median theo (year_parity, month).

    Vì sao KHÔNG dùng Prophet+LGBM cho ratio (đã thử & loại):
      - Ratio gần như KHÔNG có trend (yearly median 0.81–0.85 phẳng 10 năm)
        → Prophet trend extrapolate sai → R²=−1.39, COGS>Revenue 95% ngày.
      - Seasonal median (parity, month): R²=0.969, COGS>Rev 0% trên val 2022.
    Year-parity bắt được biennial Urban Blowout: August năm chẵn ratio≈0.79,
    năm lẻ ratio≈1.37 (clearance → biên lợi nhuận âm hợp lệ).
    """
    h = pd.DataFrame({
        "ratio":  ratio_hist.values,
        "parity": ratio_hist.index.year % 2,
        "month":  ratio_hist.index.month,
    })
    table = h.groupby(["parity", "month"])["ratio"].median()
    global_med = float(h["ratio"].median())
    return pd.Series(
        [table.get((d.year % 2, d.month), global_med) for d in index],
        index=index, name="ratio_pred",
    )


def _load_exog_regressors(index: pd.DatetimeIndex) -> dict:
    """
    Biến ngoại sinh daily-continuous cho LGBM (lag365 trong make_future_safe_features).
    Chốt sau ablation A4 (val 2022, −15.3% MAE): sessions + overstock + days_of_supply
    + fill_rate. web_traffic & inventory hết 2022 → 2024 = NaN sau lag365 (LGBM tự route).
    """
    web = pd.read_csv(WEB_FILE, parse_dates=["date"])
    sessions = web.groupby("date")["sessions"].sum().reindex(index)
    inv = pd.read_csv(INVENTORY_FILE)
    inv["snap"] = pd.to_datetime(inv["year"].astype(str) + "-" + inv["month"].astype(str) + "-01")

    def _inv_daily(col):
        return inv.groupby("snap")[col].mean().resample("D").ffill().reindex(index)

    return {
        "sessions":       sessions,
        "overstock_pct":  _inv_daily("overstock_flag"),
        "days_of_supply": _inv_daily("days_of_supply"),
        "fill_rate":      _inv_daily("fill_rate"),
    }


# ============================================================
# MAIN
# ============================================================

def run_pipeline():
    # ── Load data ─────────────────────────────────────────────
    logger.info("Loading sales data...")
    df = load_sales(SALES_FILE)
    logger.info(f"Sales: {df.index.min().date()} -> {df.index.max().date()} ({len(df)} rows)")

    # ── Bước 1: Target Denoising (no stockout impute — PH1) ───
    df = denoise_target(df)
    full_idx = pd.date_range(df.index.min(), TEST_END)

    # ── Gross margin diagnostic (train set) ───────────────────
    # Seasonal pattern của COGS/Revenue: tháng 7-8 (năm lẻ), tháng 12 có ratio > 0.95
    # hoặc > 1.0 — minh chứng vì sao không dùng hằng số, phải mô hình hóa ratio theo mùa.
    train_hist = df.loc[:TRAIN_END]
    monthly_margin = (train_hist["COGS"] / train_hist["Revenue"]).groupby(
        [train_hist.index.year, train_hist.index.month]
    ).median().unstack(level=1)
    monthly_margin.columns = [f"M{m:02d}" for m in monthly_margin.columns]
    logger.info("Gross margin (COGS/Rev) monthly median per year on train set:")
    for year, row in monthly_margin.iterrows():
        vals = "  ".join(f"{col}={v:.3f}" for col, v in row.items() if not pd.isna(v))
        logger.info(f"  {year}: {vals}")

    # ── Bước 2: Forecast Revenue (Prophet + LGBM) ─────────────
    logger.info("=" * 50)
    logger.info("BUOC 2-4: REVENUE (Prophet+LGBM) + COGS (Revenue × seasonal ratio)")
    logger.info("=" * 50)

    exog = _load_exog_regressors(full_idx)
    test_revenue_full, val_revenue = _forecast_one_target("Revenue", df, full_idx, exog)

    # ── Bước 3: COGS = Revenue × seasonal ratio (PH3) ─────────
    # Ratio dự báo bằng median (parity, month) — ratio không trend nên không cần Prophet.
    # Table ước lượng từ ≤2021 cho val 2022 (tránh leakage), từ ≤2022 cho test 2023+.
    ratio_train = (df.loc[:TRAIN_END, "COGS"] / df.loc[:TRAIN_END, "Revenue"]).clip(0.5, 1.6)
    ratio_full  = (df.loc[:VAL_END,   "COGS"] / df.loc[:VAL_END,   "Revenue"]).clip(0.5, 1.6)

    test_ratio_full = _seasonal_ratio(ratio_full, test_revenue_full.index)
    test_cogs_full  = test_revenue_full * test_ratio_full

    # COGS validation metric (cho bảng C2): Revenue_val × ratio_pred vs COGS thực 2022
    val_ratio       = _seasonal_ratio(ratio_train, val_revenue.index)
    val_cogs_pred   = (val_revenue * val_ratio).dropna()
    actual_cogs_val = df.loc[VAL_START:VAL_END, "COGS"].reindex(val_cogs_pred.index).dropna()
    val_cogs_pred   = val_cogs_pred.reindex(actual_cogs_val.index)
    cogs_mae = mean_absolute_error(actual_cogs_val, val_cogs_pred)
    cogs_r2  = r2_score(actual_cogs_val, val_cogs_pred)
    logger.info(f"  [COGS=Rev×ratio] Validation 2022: MAE={cogs_mae/1e6:.3f}M VND  R2={cogs_r2:.4f}")

    # ── Bước 5: Postprocess & Submission ─────────────────────
    logger.info("=" * 50)
    logger.info("BUOC 5: COGS DERIVATION & SUBMISSION")
    logger.info("=" * 50)

    sub = postprocess(
        revenue_pred=test_revenue_full,
        cogs_pred=test_cogs_full,
        sample_submission_path=SAMPLE_SUB_FILE,
        output_path=OUTPUT_FILE,
    )

    logger.info("THE GRIDBREAKER PIPELINE - COMPLETE")
    return sub


if __name__ == "__main__":
    run_pipeline()
