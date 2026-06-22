"""
src/lgbm_model/train_lgbm.py — Bước 3: LightGBM trên Residuals.

LightGBM học phần dư (Residual = Actual - Prophet_Prediction).
Chỉ sử dụng features 100% future-safe:
  - lag_365 / lag_364 / lag_366 của Residual
  - rolling_4w_lag365 của Residual
  - prophet_trend (component từ Prophet)
  - Exog lag365: fill_rate, days_of_supply, overstock_pct, sessions
  - Calendar: month, dayofweek, is_month_end, weekofyear, quarter, dayofyear,
              days_to_month_end, week_of_month
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
    "resid_lag730",            # M3.5: cắt compounding năm-2 (đọc 2 năm trước, thật)
    "resid_roll28_lag365",
    "prophet_trend",
    "fill_rate_lag365",        # ablation −14.2% (mạnh nhất)
    "days_of_supply_lag365",   # ablation −11.8%
    "overstock_pct_lag365",    # ablation −7.5%
    "sessions_lag365",         # ablation −4.8%
    # M3.6: exog lag730 (mỏ neo 2024→2022 thật) + is_even_year (parity promo).
    # Ablation production-sim năm-2 (feature_candidates_test.py, 3 fold): tổ hợp
    # "all 3" thắng đều năm thường −4.6% (2021 −4.5%, 2022 −4.7%), không hại COVID.
    "fill_rate_lag730", "sessions_lag730", "is_even_year",
    # M3.7: promo × parity tất định (promo_parity_test.py, walk-forward 2018-2022):
    # urban_expected + days_to_promo thắng 5/5 năm, −4.5% (ổn định mọi regime).
    "urban_expected", "days_to_promo",
    "month", "dayofweek", "is_month_end", "weekofyear", "quarter", "dayofyear",
    # Calendar nâng cao (feature_cal_backtest.py: thắng 3/3 fold, ΔMAE TB −3.8%).
    # days_to_month_end: đếm ngược liên tục tới cuối tháng (tâm lý sale cuối tháng,
    #   mượt hơn is_month_end 0/1). week_of_month: tuần 1-5 (mua sau nhận lương).
    "days_to_month_end", "week_of_month",
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

    # Lag 730 (2 năm) — CẮT COMPOUNDING năm thứ 2 (M3.5). Khi dự báo 2024,
    # lag365 đọc 2023-DỰ-ĐOÁN (nhiễu, compounding); lag730 đọc 2022-THẬT →
    # cấp tín hiệu sạch cho năm thứ 2. Production-sim năm-2: Cat+lag730 thắng
    # −4.8% vs LGBM+lag365 (production_sim_backtest.py, thắng đều 2 fold).
    X["resid_lag730"] = residuals.shift(730).reindex(idx)

    # Rolling mean 28 ngày ở lag 365
    X["resid_roll28_lag365"] = residuals.shift(365).rolling(28, min_periods=1).mean().reindex(idx)

    # Prophet trend
    X["prophet_trend"] = prophet_trend.reindex(idx)

    # Biến ngoại sinh, lag 365 (future-safe). Giá trị 2024 = NaN (web/inv hết 2022)
    # → LightGBM tự route nhánh missing.
    if extra_regressors:
        for name, series in extra_regressors.items():
            X[f"{name}_lag365"] = series.shift(365).reindex(idx)

    # Exog lag730 (M3.6): fill_rate & sessions đọc 2 NĂM trước → MỎ NEO 2024.
    # lag365 của 2024 = 2023 (web/inv HẾT 2022 → NaN); lag730 = 2022 THẬT → có data.
    # Ablation production-sim năm-2 (feature_candidates_test.py): đi CÙNG is_even_year
    # thắng đều −4.6% (sessions_lag730 đơn lẻ thua → chỉ giữ trong tổ hợp).
    if extra_regressors:
        for name in ("fill_rate", "sessions"):
            if name in extra_regressors:
                X[f"{name}_lag730"] = extra_regressors[name].shift(730).reindex(idx)

    # Calendar features
    X["month"] = idx.month
    X["dayofweek"] = idx.dayofweek
    X["is_month_end"] = idx.is_month_end.astype(int)
    X["weekofyear"] = idx.isocalendar().week.astype(int).values
    X["quarter"] = idx.quarter
    X["dayofyear"] = idx.dayofyear

    # is_even_year (M3.6): cờ năm chẵn/lẻ — phân ly quy luật promo BIENNIAL.
    # Urban Blowout/Rural chỉ năm LẺ → residual 2024 (chẵn) khác 2023 (lẻ).
    # Tất định (biết trước theo lịch) → future-safe. Ablation thắng đều 2 fold.
    X["is_even_year"] = (idx.year % 2 == 0).astype(int)

    # Calendar nâng cao (tất định → future-safe). days_to_month_end: số ngày
    # còn lại tới cuối tháng (0 nếu là ngày cuối). week_of_month: tuần thứ 1-5.
    X["days_to_month_end"] = ((idx + pd.offsets.MonthEnd(0)) - idx).days
    X["week_of_month"] = (idx.day - 1) // 7 + 1

    # Promo × parity (M3.7): khoanh REGIME promo tất định ở tầng residual.
    # urban_expected: clearance Urban Blowout (T7-T8 năm LẺ) — cú sốc biên-âm mà
    #   is_even_year + Prophet holiday chưa tách đủ sắc. Walk-forward 2018-2022:
    #   thắng 5/5 năm (gồm 2019 gãy + 2020 COVID) → ổn định nhất.
    # days_to_promo: đếm ngược (cap 60) tới family annual gần nhất (Spring doy77,
    #   MidYear 174, Fall 242, YearEnd 322) — tâm lý "sắp sale", mượt hơn cờ 0/1.
    # Tất định theo LỊCH → future-safe tuyệt đối (promotions.csv hết 2022 nhưng
    #   đây là luật, không phải lag). Tổ hợp urban+days_to: WF thắng 5/5, −4.5%.
    month = idx.month.values
    parity_odd = (idx.year.values % 2 == 1).astype(int)
    X["urban_expected"] = (((month == 7) | (month == 8)) & (parity_odd == 1)).astype(int)
    _annual_doy = np.array([77, 174, 242, 322])
    _doy = idx.dayofyear.values
    _days_to = np.min(np.abs(_doy[:, None] - _annual_doy[None, :]), axis=1)
    X["days_to_promo"] = np.clip(_days_to, 0, 60)

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


# ============================================================
# CatBoost residual learner (M3.5 — thay LightGBM ở tầng residual)
# ============================================================
# Bằng chứng (production-sim năm-2, residual_model_shootout.py):
#   CatBoost+lag730 thắng −4.8% vs LGBM+lag365, đều 2 fold; XGB/RF cũng vượt LGBM.
# Params THỦ CÔNG (depth=3) — KHÔNG Optuna-tune: tuning trên CV tĩnh overfit
# (tuning_diagnosis.py: LGBM-tuned < LGBM-default; Cat-tuned +3.0% vs manual).
# Cây nông (depth=3) + symmetric trees = "lỳ" với residual nhiễu → robust nhất
# trên năm thường 2022 (≈2024). cat_features/sample-weight/ensemble đều thử & loại.
CATBOOST_PARAMS = dict(
    iterations=300, learning_rate=0.099, depth=3,
    l2_leaf_reg=3.0, random_seed=SEED, verbose=False,
)


def fit_catboost_residual(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_name: str = "Revenue",
    params=None,
):
    """Huấn luyện CatBoost trên residuals (M3.5). Mọi feature là numerical
    (cat_features đã thử & loại — target-encoding overfit residual)."""
    from catboost import CatBoostRegressor

    logger.info(f"  Fitting CatBoost Residual cho {target_name} (M3.5, depth=3)...")
    if params is None:
        params = CATBOOST_PARAMS

    y_aligned = y_train.reindex(X_train.index)
    mask = X_train[FEAT_COLS].notna().all(axis=1) & y_aligned.notna()
    X_clean = X_train.loc[mask, FEAT_COLS]
    y_clean = y_aligned.loc[mask]

    model = CatBoostRegressor(**params)
    model.fit(X_clean, y_clean)
    logger.info(f"  CatBoost {target_name}: trained on {len(X_clean)} samples, "
                f"{len(FEAT_COLS)} features.")

    importance = pd.Series(model.get_feature_importance(), index=FEAT_COLS).sort_values(ascending=False)
    logger.info(f"  Top 5 features: {dict(importance.head().round(2).items())}")
    return model


def predict_catboost_residual(model, X: pd.DataFrame) -> pd.Series:
    """Dự đoán residual bằng CatBoost. CatBoost route NaN natively (như LGBM)."""
    n_nan = X[FEAT_COLS].isna().sum().sum()
    if n_nan > 0:
        logger.warning(f"  {n_nan} NaN cells trong X_test — giữ nguyên, CatBoost tự xử lý")
    pred = model.predict(X[FEAT_COLS])
    return pd.Series(pred, index=X.index, name="catboost_residual_pred")