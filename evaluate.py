"""
evaluate.py — Backtest mô hình CHỐT (M3.2) trên BẤT KỲ split nào

Thay đổi TRAIN_END / VAL_START / VAL_END để đo hiệu suất out-of-sample thật sự
của kiến trúc cuối cùng (Prophet+LGBM cho Revenue, COGS = Revenue × seasonal
ratio) trên bất kỳ khoảng thời gian nào → dùng cho backtest rolling-origin.

Khác với `model_comparison.py` (so 8 kiến trúc, split cố định 2022), file này
giữ NGUYÊN kiến trúc CHỐT và chỉ đổi split → kiểm tra độ ỔN ĐỊNH (robustness).

QUAN TRỌNG: Để đánh giá out-of-sample thật, cần TRAIN_END < VAL_START.

Ví dụ rolling-origin (chạy lần lượt, đổi 3 dòng cấu hình bên dưới):
    Fold A: TRAIN_END=2019-12-31, VAL_START=2020-01-01, VAL_END=2020-12-31
    Fold B: TRAIN_END=2020-12-31, VAL_START=2021-01-01, VAL_END=2021-12-31
    Fold C: TRAIN_END=2021-12-31, VAL_START=2022-01-01, VAL_END=2022-12-31

Split nhiều năm (vd val 2021–2022) được xử lý bằng rolling forecast: năm sau
dùng PROXY residual (dự báo) của năm trước cho lag365 — KHÔNG nhìn nhãn thật.

Usage:
    python evaluate.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import logging
for noisy in ("prophet", "cmdstanpy", "gridbreaker"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

from src.utils import load_sales, mape, SEED
from src.denoising import denoise_target
from src.prophet_model import fit_prophet, predict_prophet
from src.lgbm_model import make_future_safe_features
from src.lgbm_model.train_lgbm import FEAT_COLS
from src.postprocess import blend_forecasts
from pipeline import _seasonal_ratio, _load_exog_regressors

np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════
# ✏️  THAY ĐỔI Ở ĐÂY để thử các split khác nhau
# ══════════════════════════════════════════════════════════════
TRAIN_END = "2020-12-31"   # Mô hình học đến ngày này
VAL_START = "2021-01-01"   # Bắt đầu đánh giá (mô hình chưa thấy)
VAL_END   = "2022-12-31"   # Kết thúc đánh giá
# ══════════════════════════════════════════════════════════════

# Siêu tham số LGBM tốt nhất (Optuna tìm 1 lần trong pipeline.py). Dùng cố định
# cho mọi fold để kiểm tra độ ỔN ĐỊNH của kiến trúc, không re-tune mỗi fold.
LGB_PARAMS = dict(learning_rate=0.099, max_depth=3, num_leaves=41,
                  min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=8.37, reg_lambda=1.65, n_estimators=300,
                  random_state=SEED, verbose=-1)

CSV_DIR    = os.path.join(ROOT, "csv")
SALES_FILE = os.path.join(CSV_DIR, "sales.csv")


def validate_split(train_end: str, val_start: str, val_end: str):
    """Kiểm tra tính hợp lệ của split."""
    t, vs, ve = map(pd.Timestamp, (train_end, val_start, val_end))
    if vs <= t:
        print(f"[!] CANH BAO: VAL_START ({val_start}) nam TRONG/TRUOC TRAIN_END ({train_end})")
        print("    -> Day la IN-SAMPLE evaluation (metrics lac quan hon thuc te)")
    else:
        print(f"[OK] OUT-OF-SAMPLE: gap = {(vs - t).days} ngay giua train va val")
    print(f"   Train: data_start → {train_end}")
    print(f"   Val  : {val_start} → {val_end} ({(ve - vs).days + 1} ngày)")


def forecast_revenue(df, train_end, val_start, val_end, full_idx, exog):
    """Prophet + LGBM-residual với rolling forecast qua từng năm val.

    Năm đầu val dùng residual train (đã biết); các năm sau dùng PROXY residual
    (dự báo) của năm trước → OOS thật, không nhìn nhãn val.
    """
    train_series = df.loc[:train_end, "Clean_Revenue"].dropna()
    mp = fit_prophet(train_series, target_name="Revenue")
    pp = predict_prophet(mp, str(df.index.min().date()), val_end)

    train_prophet  = pp.loc[:train_end, "prophet_pred"]
    residual_train = df.loc[:train_end, "Revenue"].reindex(train_prophet.index) - train_prophet
    full_residuals = pd.Series(np.nan, index=full_idx, name="residual")
    full_residuals.update(residual_train)
    full_trend = pp["prophet_trend"].reindex(full_idx)

    # Fit LGBM 1 lần trên train
    X_full   = make_future_safe_features(full_residuals, full_trend, full_idx, exog)
    X_tr     = X_full.loc[:train_end, FEAT_COLS]
    y_tr     = residual_train.reindex(X_tr.index)
    mask     = X_tr.notna().all(axis=1) & y_tr.notna()
    model    = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr.loc[mask], y_tr.loc[mask])

    # Rolling predict qua từng năm val
    residuals_rolling = full_residuals.copy()
    preds = []
    for year in range(pd.Timestamp(val_start).year, pd.Timestamp(val_end).year + 1):
        y_start = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(val_start))
        y_end   = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(val_end))
        Xi      = make_future_safe_features(residuals_rolling, full_trend, full_idx, exog)
        Xyr     = Xi.loc[y_start:y_end, FEAT_COLS]
        pr_yr   = pp.loc[y_start:y_end, "prophet_pred"]
        resid_yr = pd.Series(model.predict(Xyr), index=Xyr.index)
        pred_yr  = blend_forecasts(pr_yr, resid_yr)
        preds.append(pred_yr)
        # Feed proxy residual cho lag365 năm sau (KHÔNG dùng nhãn thật)
        residuals_rolling.update(pred_yr - pr_yr)

    return pd.concat(preds)


def run_evaluation(train_end: str, val_start: str, val_end: str):
    print("=" * 60)
    print("  THE GRIDBREAKER — EVALUATION (kiến trúc CHỐT M3.2)")
    print("=" * 60)
    validate_split(train_end, val_start, val_end)
    print()

    print("[1/3] Load & Denoise (no stockout impute — PH1)...")
    df = denoise_target(load_sales(SALES_FILE))
    full_idx = pd.date_range(df.index.min(), val_end)
    if len(df.loc[:train_end]) < 365:
        print(f"[ERR] Train chi co {len(df.loc[:train_end])} ngay, can >= 365 cho lag_365.")
        return
    exog = _load_exog_regressors(full_idx)

    print("[2/3] Forecast Revenue (Prophet+LGBM, rolling) ...")
    rev_pred = forecast_revenue(df, train_end, val_start, val_end, full_idx, exog)

    print("[3/3] COGS = Revenue × seasonal ratio (parity, month) ...")
    # Ratio table chỉ từ ≤train_end → tránh leakage
    ratio_train = (df.loc[:train_end, "COGS"] / df.loc[:train_end, "Revenue"]).clip(0.5, 1.6)
    cogs_pred = rev_pred * _seasonal_ratio(ratio_train, rev_pred.index)

    results = {"Revenue": rev_pred, "COGS": cogs_pred}

    print("\n" + "=" * 60)
    print("  KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 60)
    print(f"  Split: Train → {train_end} | Val: {val_start} → {val_end}\n")

    summary_rows = []
    for target in ["Revenue", "COGS"]:
        actual = df.loc[val_start:val_end, target].reindex(results[target].index).dropna()
        pred   = results[target].reindex(actual.index)
        mae_v  = mean_absolute_error(actual, pred)
        rmse_v = np.sqrt(mean_squared_error(actual, pred))
        r2_v   = r2_score(actual, pred)
        mape_v = mape(actual.values, pred.values)
        print(f"  {target}:  MAE={mae_v/1e6:.3f}M  RMSE={rmse_v/1e6:.3f}M  "
              f"R²={r2_v:.4f}  MAPE={mape_v:.2f}%")
        summary_rows.append({"Target": target, "MAE (M)": round(mae_v/1e6, 3),
                             "RMSE (M)": round(rmse_v/1e6, 3), "R²": round(r2_v, 4),
                             "MAPE (%)": round(mape_v, 2)})

    # COGS > Revenue audit (kiến trúc ratio: chỉ tháng 8 năm lẻ → hợp lệ)
    cj = pd.concat([rev_pred.rename("r"), cogs_pred.rename("c")], axis=1).dropna()
    pct_clip = float((cj["c"] > cj["r"]).mean() * 100)
    print(f"\n  COGS > Revenue: {pct_clip:.1f}% số ngày "
          f"({'OK — clearance năm lẻ' if pct_clip < 20 else '⚠️ cao bất thường'})")

    print("\n" + "─" * 60)
    print("  SO SÁNH VỚI BASELINE (YoY Naive):")
    for target in ["Revenue", "COGS"]:
        actual   = df.loc[val_start:val_end, target].reindex(results[target].index).dropna()
        baseline = df[target].shift(365).reindex(actual.index)
        m        = baseline.notna()
        if m.sum() > 0:
            bl_mae = mean_absolute_error(actual[m], baseline[m])
            gb_mae = mean_absolute_error(actual[m], results[target].reindex(actual.index)[m])
            print(f"  {target}: Baseline MAE={bl_mae/1e6:.3f}M → "
                  f"CHỐT cải thiện {(bl_mae - gb_mae)/bl_mae*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation(TRAIN_END, VAL_START, VAL_END)
