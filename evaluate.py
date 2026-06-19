"""
evaluate.py — Kiểm tra sai số mô hình với bất kỳ split nào

Thay đổi TRAIN_END / VAL_START / VAL_END để kiểm tra hiệu suất thật sự
(out-of-sample) của mô hình trên bất kỳ khoảng thời gian nào.

QUAN TRỌNG: Để đánh giá out-of-sample thật sự, cần đảm bảo:
    TRAIN_END < VAL_START

Ví dụ các split hợp lệ:
    # Split 1: Train 2012-2019, Val 2020-2022 (3 năm holdout)
    TRAIN_END = "2019-12-31"
    VAL_START = "2020-01-01"
    VAL_END   = "2022-12-31"

    # Split 2: Train 2012-2020, Val 2021-2022 (2 năm holdout)
    TRAIN_END = "2020-12-31"
    VAL_START = "2021-01-01"
    VAL_END   = "2022-12-31"

    # Split 3: Train 2012-2021, Val 2022 (1 năm holdout)
    TRAIN_END = "2021-12-31"
    VAL_START = "2022-01-01"
    VAL_END   = "2022-12-31"

Usage:
    python evaluate.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.utils import load_sales, load_inventory_flags, mape, SEED
from src.denoising import denoise_target
from src.prophet_model import fit_prophet, predict_prophet
from src.lgbm_model import fit_lgbm_residual, predict_lgbm_residual, make_future_safe_features
from src.postprocess import blend_forecasts

np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════
# ✏️  THAY ĐỔI Ở ĐÂY để thử các split khác nhau
# ══════════════════════════════════════════════════════════════
TRAIN_END = "2020-12-31"   # Mô hình học đến ngày này
VAL_START = "2021-01-01"   # Bắt đầu đánh giá (mô hình chưa thấy)
VAL_END   = "2022-12-31"   # Kết thúc đánh giá
# ══════════════════════════════════════════════════════════════

CSV_DIR        = os.path.join(ROOT, "csv")
SALES_FILE     = os.path.join(CSV_DIR, "sales.csv")
INVENTORY_FILE = os.path.join(CSV_DIR, "inventory.csv")


def validate_split(train_end: str, val_start: str, val_end: str):
    """Kiểm tra tính hợp lệ của split."""
    t = pd.Timestamp(train_end)
    vs = pd.Timestamp(val_start)
    ve = pd.Timestamp(val_end)
    if vs <= t:
        print(f"[!] CANH BAO: VAL_START ({val_start}) nam TRONG hoac TRUOC TRAIN_END ({train_end})")
        print("    -> Day la IN-SAMPLE evaluation (metrics lac quan hon thuc te)")
    else:
        gap = (vs - t).days
        print(f"[OK] OUT-OF-SAMPLE evaluation: gap = {gap} ngay giua train va val")
    print(f"   Train: data_start → {train_end}")
    print(f"   Val  : {val_start} → {val_end} ({(ve - vs).days + 1} ngày)")


def run_evaluation(train_end: str, val_start: str, val_end: str):
    print("=" * 60)
    print("  THE GRIDBREAKER — EVALUATION MODE")
    print("=" * 60)

    validate_split(train_end, val_start, val_end)
    print()

    # Load & Denoise
    print("[1/4] Load & Denoise dữ liệu...")
    df = load_sales(SALES_FILE)
    stockout_flags = load_inventory_flags(INVENTORY_FILE)
    df = denoise_target(df, stockout_flags.get("stockout_flag", None))

    # Chỉ dùng data đến val_end để build full_idx
    full_idx = pd.date_range(df.index.min(), val_end)
    train = df.loc[:train_end]

    if len(train) < 365:
        print(f"[ERR] Train set chi co {len(train)} ngay, can >= 365 ngay cho lag_365.")
        return

    print(f"   Train: {train.index.min().date()} → {train.index.max().date()} ({len(train)} ngày)")
    print(f"   Val  : {val_start} → {val_end}")
    print()

    results = {}
    for target in ["Revenue", "COGS"]:
        print(f"[2-3/4] Prophet + LightGBM cho {target}...")
        clean_col = f"Clean_{target}"
        train_series = train[clean_col].dropna()

        # Prophet
        prophet_model = fit_prophet(train_series, target_name=target)
        prophet_preds = predict_prophet(prophet_model, str(df.index.min().date()), val_end)

        # Residual
        train_prophet = prophet_preds.loc[:train_end, "prophet_pred"]
        residual_train = train[target].reindex(train_prophet.index) - train_prophet

        full_residuals = pd.Series(np.nan, index=full_idx, name="residual")
        full_residuals.update(residual_train)
        full_trend = prophet_preds["prophet_trend"].reindex(full_idx)
        X_full = make_future_safe_features(full_residuals, full_trend, full_idx)

        # LightGBM — chỉ train đến TRAIN_END
        X_train = X_full.loc[:train_end]
        y_train = residual_train.reindex(X_train.index)
        lgbm_model = fit_lgbm_residual(X_train, y_train, target_name=target)

        # Predict val
        val_prophet = prophet_preds.loc[val_start:val_end, "prophet_pred"]
        X_val = X_full.loc[val_start:val_end]
        val_resid = predict_lgbm_residual(lgbm_model, X_val)
        val_final = blend_forecasts(val_prophet, val_resid)

        actual = df.loc[val_start:val_end, target].reindex(val_final.index).dropna()
        predicted = val_final.reindex(actual.index)

        results[target] = {
            "actual":    actual,
            "predicted": predicted,
        }

    # Metrics
    print()
    print("=" * 60)
    print("  KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 60)
    print(f"  Split: Train → {train_end} | Val: {val_start} → {val_end}")
    print()

    summary_rows = []
    for target in ["Revenue", "COGS"]:
        actual    = results[target]["actual"]
        predicted = results[target]["predicted"]

        mae_v  = mean_absolute_error(actual, predicted)
        rmse_v = np.sqrt(mean_squared_error(actual, predicted))
        r2_v   = r2_score(actual, predicted)
        mape_v = mape(actual.values, predicted.values)

        print(f"  {target}:")
        print(f"    MAE  = {mae_v/1e6:.3f}M VND")
        print(f"    RMSE = {rmse_v/1e6:.3f}M VND")
        print(f"    R²   = {r2_v:.4f}")
        print(f"    MAPE = {mape_v:.2f}%")
        print()

        summary_rows.append({
            "Target": target, "MAE (M)": round(mae_v/1e6, 3),
            "RMSE (M)": round(rmse_v/1e6, 3), "R²": round(r2_v, 4),
            "MAPE (%)": round(mape_v, 2),
        })

    summary = pd.DataFrame(summary_rows).set_index("Target")
    print(summary.to_string())
    print()

    # So sánh với Baseline (YoY Naive)
    print("─" * 60)
    print("  SO SÁNH VỚI BASELINE (YoY Naive):")
    for target in ["Revenue", "COGS"]:
        actual    = results[target]["actual"]
        baseline  = df[target].shift(365).loc[val_start:val_end].reindex(actual.index)
        mask      = baseline.notna()
        if mask.sum() > 0:
            bl_mae  = mean_absolute_error(actual[mask], baseline[mask])
            bl_r2   = r2_score(actual[mask], baseline[mask])
            gb_mae  = mean_absolute_error(actual[mask], results[target]["predicted"][mask])
            improve = (bl_mae - gb_mae) / bl_mae * 100
            print(f"  {target}: Baseline MAE={bl_mae/1e6:.3f}M (R²={bl_r2:.3f}) "
                  f"→ Gridbreaker cải thiện {improve:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation(TRAIN_END, VAL_START, VAL_END)
