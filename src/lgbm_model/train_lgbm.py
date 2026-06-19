"""
src/lgbm_model/train_lgbm.py — Bước 3: LightGBM trên Residuals.

LightGBM học phần dư (Residual = Actual - Prophet_Prediction).
Chỉ sử dụng features 100% future-safe:
  - lag_365 / lag_364 / lag_366 của Residual
  - rolling_4w_lag365 của Residual
  - prophet_trend (component từ Prophet)
  - Calendar: month, dayofweek, is_month_end, weekofyear, quarter
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import logging

logger = logging.getLogger("gridbreaker")

SEED = 42
EARLY_STOPPING_ROUNDS = 50
# Feature set chốt sau ablation val 2022 (A3/A4). Biến inv+sess (lag365, future-safe)
# giảm MAE −15.3% so với base. fill_rate/days_of_supply mạnh nhất; promo để Prophet
# event xử lý (promo_lag730 trong LGBM làm TỆ hơn +1.8%).
FEAT_COLS = [
    "resid_lag365", "resid_lag364", "resid_lag366",
    "resid_roll28_lag365",
    "prophet_trend",
    "fill_rate_lag365",        # ablation −14.2% (mạnh nhất)
    "days_of_supply_lag365",   # ablation −11.8%
    "overstock_pct_lag365",    # ablation −7.5%
    "sessions_lag365",         # ablation −4.8%
    "month", "dayofweek", "is_month_end", "weekofyear", "quarter", "dayofyear",
]


def make_future_safe_features(
    residuals: pd.Series,
    prophet_trend: pd.Series,
    index: pd.DatetimeIndex = None,
    extra_regressors: dict = None,
) -> pd.DataFrame:
    """
    Tạo feature matrix 100% future-safe từ residuals và prophet trend.

    Args:
        residuals: Series chứa Residual (index=Date).
                   Có thể chứa NaN cho kỳ test.
        prophet_trend: Series chứa trend component từ Prophet.
        index: DatetimeIndex cuối cùng (nếu None, dùng residuals.index).
        extra_regressors: dict {name: Series} — biến ngoại sinh (sessions, overstock_pct).
                   Mỗi biến được shift(365) → future-safe (chỉ dùng giá trị năm trước).
                   Series phải ở daily-continuous index để shift(365) = đúng 365 ngày.
    """
    idx = index if index is not None else residuals.index
    X = pd.DataFrame(index=idx)

    # Lag features (từ residuals — lag 365 luôn có trong train data)
    for lag in [365, 364, 366]:
        X[f"resid_lag{lag}"] = residuals.shift(lag).reindex(idx)

    # Rolling mean 28 ngày ở lag 365
    X["resid_roll28_lag365"] = residuals.shift(365).rolling(28, min_periods=1).mean().reindex(idx)

    # Prophet trend
    X["prophet_trend"] = prophet_trend.reindex(idx)

    # Biến ngoại sinh, lag 365 (future-safe). Giá trị 2024 = NaN (web/inv hết 2022)
    # → LightGBM tự route nhánh missing.
    if extra_regressors:
        for name, series in extra_regressors.items():
            X[f"{name}_lag365"] = series.shift(365).reindex(idx)

    # Calendar features
    X["month"] = idx.month
    X["dayofweek"] = idx.dayofweek
    X["is_month_end"] = idx.is_month_end.astype(int)
    X["weekofyear"] = idx.isocalendar().week.astype(int).values
    X["quarter"] = idx.quarter
    X["dayofyear"] = idx.dayofyear

    return X


def fit_lgbm_residual(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_name: str = "Revenue",
    params = None,
    eval_set=None
) -> lgb.LGBMRegressor:
    """Huấn luyện LightGBM trên residuals."""
    logger.info(f"  Fitting LightGBM Residual cho {target_name}...")

    if params is None:
        params = {
            "n_estimators":      300,
            "learning_rate":     0.05,
            "max_depth":         5,
            "num_leaves":        31,
            "min_child_samples": 30,
            "subsample":         0.8,
            "colsample_bytree":  0.8,
            "reg_alpha":         0.1,
            "reg_lambda":        1.0,
            "random_state":      SEED,
            "verbose":           -1,
        }

    model = lgb.LGBMRegressor(**params)

    # Drop NaN rows (lag_365 cần >= 1 năm data).
    # reindex y_train về index của X_train trước khi notna(): nếu hai Series
    # lệch index, phép `&` giữa hai boolean Series sẽ align theo union và có
    # thể sinh NaN ở phần không khớp, làm mask sai âm thầm.
    y_aligned = y_train.reindex(X_train.index)
    mask = X_train[FEAT_COLS].notna().all(axis=1) & y_aligned.notna()
    X_clean = X_train.loc[mask, FEAT_COLS]
    y_clean = y_aligned.loc[mask]

    fit_kwargs = {}
    if eval_set is not None:
        X_es, y_es = eval_set
        fit_kwargs["eval_set"] = [(X_es[FEAT_COLS], y_es)]
        # eval_metric phải khớp với metric dùng để chọn best_params ở Optuna
        # (MAE). Không set thì LightGBM mặc định early-stop theo L2, có thể
        # chọn sai iteration so với cái Optuna thực sự đang tối ưu.
        fit_kwargs["eval_metric"] = "mae"
        fit_kwargs["callbacks"] = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]

    model.fit(X_clean, y_clean, **fit_kwargs)
    logger.info(f"  LightGBM {target_name}: trained on {len(X_clean)} samples, {len(FEAT_COLS)} features.")

    # Log top features
    importance = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
    logger.info(f"  Top 5 features: {dict(importance.head().items())}")

    return model


def predict_lgbm_residual(
    model: lgb.LGBMRegressor,
    X: pd.DataFrame,
) -> pd.Series:
    """Dự đoán residual bằng LightGBM."""
    n_nan = X[FEAT_COLS].isna().sum().sum()
    if n_nan > 0:
        # KHÔNG fillna(0): LightGBM xử lý missing value natively bằng cách học
        # một hướng rẽ nhánh riêng cho NaN ngay trong lúc train. fillna(0) sẽ
        # biến "không có dữ liệu" thành "giá trị thực = 0" — sai đặc biệt với
        # prophet_trend (không bao giờ thực sự gần 0), khiến model coi các
        # hàng thiếu dữ liệu là outlier ở đáy phân phối thay vì route đúng
        # theo nhánh missing đã học.
        logger.warning(f"  {n_nan} NaN cells trong X_test — giữ nguyên NaN, để LightGBM tự xử lý")
    pred = model.predict(X[FEAT_COLS])
    return pd.Series(pred, index=X.index, name="lgbm_residual_pred")