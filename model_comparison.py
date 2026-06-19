"""
model_comparison.py — SO SÁNH 8 KIẾN TRÚC MÔ HÌNH (đối chiếu GIAI_THICH §7-9)
=============================================================================
Chạy lần lượt 8 kiến trúc trên CÙNG một split (train ≤2021 / val 2022 OOS) và
in bảng so sánh MAE / RMSE / R² (Revenue) + COGS R² + % ngày COGS>Revenue.

    M1.0  YoY Naive              — pred = Revenue 365 ngày trước
    M1.1  ARIMA(p,d,q)           — statistical baseline (statsmodels)
    M2.0  Prophet đơn            — chỉ trend+season, không LGBM
    M2.1  Prophet+LGBM, COGS độc lập      — 2 model COGS/Rev rời → COGS trôi
    M2.2  Prophet+LGBM, COGS ratio-qua-Prophet — ratio không trend → hỏng
    M3.0  Hybrid + Seasonal Ratio COGS    — Rev=Prophet(no promo)+LGBM base
    M3.1  + Promo Events tất định         — Prophet thêm 6 promo family
    M3.2  + 4 Feature tồn kho/web ✅ CHỐT  — = pipeline.py

GHI CHÚ PHƯƠNG PHÁP:
  - Các biến thể Hybrid (M3.0–M3.2) dùng CHUNG bộ siêu tham số LGBM tốt nhất
    (Optuna tìm 1 lần trong pipeline.py) để CÔ LẬP ảnh hưởng KIẾN TRÚC, không
    để biến động Optuna nhiễu kết quả. M3.2 ở đây ≈ pipeline.py (nguồn sự thật).
  - M2.1/M2.2 là kiến trúc ĐÃ LOẠI, dựng lại trên dữ liệu hiện tại để minh chứng
    thất bại (COGS>Revenue cao) — số có thể lệch log lịch sử (denoise đã đổi).
  - Đánh giá trên val 2022 (năm CHẴN, không có Urban Blowout) → pct_clipped của
    M3.x ≈ 0%. Con số 5.7% trong báo cáo là trên TEST 2023–24 (năm lẻ, clearance
    tháng 8 hợp lệ) — xem pipeline.py.

Chạy:  python model_comparison.py
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, ".")
from prophet import Prophet
from src.utils import load_sales
from src.denoising import denoise_target
from src.prophet_model.train_prophet import _build_holidays
from src.lgbm_model import make_future_safe_features
from pipeline import _seasonal_ratio, _load_exog_regressors

import logging
for noisy in ("prophet", "cmdstanpy", "gridbreaker"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

CSV = "csv"
TRAIN_END, VAL_START, VAL_END = "2021-12-31", "2022-01-01", "2022-12-31"

# Siêu tham số LGBM tốt nhất (Optuna TPE 50 trials, tìm 1 lần trong pipeline.py).
BEST_PARAMS = dict(learning_rate=0.099, max_depth=3, num_leaves=41,
                   min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=8.37, reg_lambda=1.65, n_estimators=1000,
                   random_state=42, verbose=-1)

BASE_FEATS = ["resid_lag365", "resid_lag364", "resid_lag366", "resid_roll28_lag365",
              "prophet_trend", "month", "dayofweek", "is_month_end",
              "weekofyear", "quarter", "dayofyear"]
EXOG_FEATS = ["fill_rate_lag365", "days_of_supply_lag365",
              "overstock_pct_lag365", "sessions_lag365"]
FULL_FEATS = BASE_FEATS + EXOG_FEATS


# ──────────────────────────────────────────────────────────────────────────
# Khối dựng sẵn
# ──────────────────────────────────────────────────────────────────────────
def fit_prophet_cfg(series: pd.Series, use_promo: bool) -> Prophet:
    """Prophet với holiday bật/tắt 6 promo family (tách M3.0 vs M3.1)."""
    hol = _build_holidays()
    if not use_promo:
        hol = hol[~hol["holiday"].str.startswith("promo_")]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                holidays=hol, changepoint_prior_scale=0.15, seasonality_prior_scale=10.0,
                holidays_prior_scale=10.0, changepoint_range=0.9)
    m.fit(pd.DataFrame({"ds": series.index, "y": series.values}))
    return m


def prophet_predict(model: Prophet, start: str, end: str) -> pd.DataFrame:
    fut = model.make_future_dataframe(
        periods=(pd.Timestamp(end) - model.history["ds"].max()).days, freq="D")
    fc = model.predict(fut).set_index("ds")
    return fc.loc[start:end, ["yhat", "trend"]].rename(
        columns={"yhat": "prophet_pred", "trend": "prophet_trend"})


def hybrid_val_pred(df, target, use_promo, feats, exog, full_idx):
    """Prophet(+promo?) + LGBM-residual → dự báo val 2022 (không cần rolling vì
    lag365 của 2022 = residual 2021 đã biết). Trả (val_pred, prophet_df)."""
    train_series = df.loc[:TRAIN_END, f"Clean_{target}"].dropna()
    mp = fit_prophet_cfg(train_series, use_promo)
    pp = prophet_predict(mp, str(df.index.min().date()), VAL_END)

    train_prophet = pp.loc[:TRAIN_END, "prophet_pred"]
    resid_train = df.loc[:TRAIN_END, target].reindex(train_prophet.index) - train_prophet
    full_resid = pd.Series(np.nan, index=full_idx); full_resid.update(resid_train)
    trend = pp["prophet_trend"].reindex(full_idx)

    X = make_future_safe_features(full_resid, trend, full_idx, exog)
    Xtr = X.loc[:TRAIN_END, feats].dropna()
    ytr = resid_train.reindex(Xtr.index).dropna()
    Xtr = Xtr.reindex(ytr.index)
    m = lgb.LGBMRegressor(**BEST_PARAMS)
    m.fit(Xtr, ytr)

    val_prophet = pp.loc[VAL_START:VAL_END, "prophet_pred"]
    Xval = X.loc[VAL_START:VAL_END, feats]
    val_pred = val_prophet + pd.Series(m.predict(Xval), index=Xval.index)
    return val_pred, pp


def metrics(actual: pd.Series, pred: pd.Series):
    j = pd.concat([actual.rename("a"), pred.rename("p")], axis=1).dropna()
    a, p = j["a"], j["p"]
    return (mean_absolute_error(a, p) / 1e6,
            np.sqrt(mean_squared_error(a, p)) / 1e6,
            r2_score(a, p))


# ──────────────────────────────────────────────────────────────────────────
# 8 kiến trúc
# ──────────────────────────────────────────────────────────────────────────
def run_all():
    df = denoise_target(load_sales(f"{CSV}/sales.csv"))
    full_idx = pd.date_range(df.index.min(), VAL_END)
    exog = _load_exog_regressors(full_idx)

    rev_val = df.loc[VAL_START:VAL_END, "Revenue"]
    cogs_val = df.loc[VAL_START:VAL_END, "COGS"]
    ratio_train = (df.loc[:TRAIN_END, "COGS"] / df.loc[:TRAIN_END, "Revenue"]).clip(0.5, 1.6)

    rows = []   # mỗi dòng: dict số liệu 1 model

    def rec(mid, name, rev_pred=None, cogs_pred=None, note=""):
        d = {"id": mid, "name": name, "note": note}
        if rev_pred is not None:
            d["rev_mae"], d["rev_rmse"], d["rev_r2"] = metrics(rev_val, rev_pred)
        if cogs_pred is not None:
            cj = pd.concat([cogs_val.rename("a"), cogs_pred.rename("c"),
                            (rev_pred if rev_pred is not None else rev_val).rename("r")],
                           axis=1).dropna()
            d["cogs_mae"] = mean_absolute_error(cj["a"], cj["c"]) / 1e6
            d["cogs_r2"] = r2_score(cj["a"], cj["c"])
            d["pct_clip"] = float((cj["c"] > cj["r"]).mean() * 100)
        rows.append(d)
        print(f"  [{mid}] {name} ... done")

    print("Đang chạy 8 kiến trúc (val 2022 OOS)...")

    # M1.0 — YoY Naive
    yoy = df["Revenue"].shift(365).loc[VAL_START:VAL_END]
    rec("M1.0", "YoY Naive", rev_pred=yoy, note="baseline ngây thơ")

    # M1.1 — ARIMA(p,d,q) statistical baseline
    try:
        from statsmodels.tsa.arima.model import ARIMA
        y_tr = df.loc[:TRAIN_END, "Revenue"].asfreq("D").interpolate()
        order = (5, 1, 2)
        res = ARIMA(y_tr, order=order).fit()
        steps = len(pd.date_range(VAL_START, VAL_END))
        fc = res.forecast(steps=steps)
        fc.index = pd.date_range(VAL_START, VAL_END)
        rec("M1.1", f"ARIMA{order}", rev_pred=fc, note="không bắt được mùa vụ năm")
    except Exception as e:
        print(f"  [M1.1] ARIMA lỗi: {e}")
        rows.append({"id": "M1.1", "name": "ARIMA", "note": f"lỗi: {e}"})

    # M2.0 — Prophet đơn (no promo, no LGBM)
    mp = fit_prophet_cfg(df.loc[:TRAIN_END, "Clean_Revenue"].dropna(), use_promo=False)
    pp = prophet_predict(mp, VAL_START, VAL_END)
    rec("M2.0", "Prophet đơn", rev_pred=pp["prophet_pred"], note="chỉ trend+season")

    # M2.1 — Prophet+LGBM Revenue + COGS ĐỘC LẬP (2 model rời)
    rev21, _ = hybrid_val_pred(df, "Revenue", False, BASE_FEATS, None, full_idx)
    cogs21, _ = hybrid_val_pred(df, "COGS", False, BASE_FEATS, None, full_idx)
    rec("M2.1", "Prophet+LGBM, COGS độc lập", rev_pred=rev21, cogs_pred=cogs21,
        note="2 model rời → COGS trôi khỏi Revenue")

    # M2.2 — COGS = Rev × ratio-qua-Prophet (ratio không trend → hỏng)
    ratio_series = (df.loc[:TRAIN_END, "COGS"] / df.loc[:TRAIN_END, "Revenue"]).clip(0.5, 1.6)
    mr = fit_prophet_cfg(ratio_series, use_promo=False)
    ratio_pred22 = prophet_predict(mr, VAL_START, VAL_END)["prophet_pred"]
    cogs22 = rev21 * ratio_pred22.reindex(rev21.index)
    rec("M2.2", "Prophet+LGBM, COGS ratio-qua-Prophet", rev_pred=rev21, cogs_pred=cogs22,
        note="ratio phẳng → Prophet extrapolate sai")

    # M3.0 — Hybrid + Seasonal Ratio (no promo, base feats)
    rev30, _ = hybrid_val_pred(df, "Revenue", False, BASE_FEATS, None, full_idx)
    cogs30 = rev30 * _seasonal_ratio(ratio_train, rev30.index)
    rec("M3.0", "Hybrid + Seasonal Ratio COGS", rev_pred=rev30, cogs_pred=cogs30,
        note="COGS=Rev×median_ratio(parity,month)")

    # M3.1 — + Promo Events (prophet WITH promo, base feats)
    rev31, _ = hybrid_val_pred(df, "Revenue", True, BASE_FEATS, None, full_idx)
    cogs31 = rev31 * _seasonal_ratio(ratio_train, rev31.index)
    rec("M3.1", "+ Promo Events tất định", rev_pred=rev31, cogs_pred=cogs31,
        note="Prophet + 6 promo family")

    # M3.2 — + 4 exog (CHỐT) = pipeline.py
    rev32, _ = hybrid_val_pred(df, "Revenue", True, FULL_FEATS, exog, full_idx)
    cogs32 = rev32 * _seasonal_ratio(ratio_train, rev32.index)
    rec("M3.2", "+ 4 Feature tồn kho/web ✅ CHỐT", rev_pred=rev32, cogs_pred=cogs32,
        note="fill_rate+days_supply+overstock+sessions lag365")

    return rows


def print_table(rows):
    print("\n" + "=" * 100)
    print("SO SÁNH 8 KIẾN TRÚC — Validation OOS 2022 (train ≤2021)")
    print("=" * 100)
    h = f"{'ID':5s} {'Mô hình':34s} {'Rev MAE':>8s} {'Rev RMSE':>9s} {'Rev R²':>8s} " \
        f"{'COGS R²':>8s} {'COGS>Rev%':>10s}"
    print(h)
    print("-" * 100)
    for d in rows:
        rev_mae = f"{d['rev_mae']:.3f}" if "rev_mae" in d else "—"
        rev_rmse = f"{d['rev_rmse']:.3f}" if "rev_rmse" in d else "—"
        rev_r2 = f"{d['rev_r2']:.3f}" if "rev_r2" in d else "—"
        cogs_r2 = f"{d['cogs_r2']:.3f}" if "cogs_r2" in d else "—"
        clip = f"{d['pct_clip']:.1f}%" if "pct_clip" in d else "—"
        print(f"{d['id']:5s} {d['name'][:34]:34s} {rev_mae:>8s} {rev_rmse:>9s} "
              f"{rev_r2:>8s} {cogs_r2:>8s} {clip:>10s}")
    print("=" * 100)
    print("Ghi chú: Rev MAE/RMSE đơn vị triệu VND. pct COGS>Rev trên val 2022 (năm chẵn)")
    print("→ M3.x ≈ 0%; trên TEST 2023–24 (năm lẻ, clearance tháng 8) M3.2 = 5.7% (pipeline.py).")

    # Lưu CSV làm minh chứng
    out = pd.DataFrame(rows).set_index("id")
    out.to_csv("model_comparison.csv", encoding="utf-8")
    print("\nĐã lưu: model_comparison.csv")


if __name__ == "__main__":
    print("#" * 100)
    print("#  SO SÁNH 8 KIẾN TRÚC MÔ HÌNH — The Gridbreaker (GIAI_THICH §7-9)")
    print("#" * 100)
    rows = run_all()
    print_table(rows)
