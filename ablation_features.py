"""
ablation_features.py — A3/A4: chọn feature cho LGBM residual (Revenue) bằng
backtest val 2022. Fit Prophet 1 lần (đã có promo events), giữ LGBM params cố
định để cô lập ảnh hưởng feature. KHÔNG phải pipeline chính — chạy 1 lần để
quyết FEAT_COLS, rồi có thể xóa.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, ".")
from src.utils import load_sales
from src.denoising import denoise_target
from src.prophet_model import fit_prophet, predict_prophet
from src.lgbm_model import make_future_safe_features

TRAIN_END, VAL_START, VAL_END = "2021-12-31", "2022-01-01", "2022-12-31"
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=5, num_leaves=31,
                  min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)

# ---- data ----
df = denoise_target(load_sales("csv/sales.csv"))
full_idx = pd.date_range(df.index.min(), VAL_END)

# ---- exogenous daily series ----
web = pd.read_csv("csv/web_traffic.csv", parse_dates=["date"])
sessions = web.groupby("date")["sessions"].sum().reindex(full_idx)

inv = pd.read_csv("csv/inventory.csv")
inv["snap"] = pd.to_datetime(inv["year"].astype(str) + "-" + inv["month"].astype(str) + "-01")
def inv_daily(col):
    return inv.groupby("snap")[col].mean().resample("D").ffill().reindex(full_idx)
overstock = inv_daily("overstock_flag")
days_supply = inv_daily("days_of_supply")
fill_rate = inv_daily("fill_rate")

# deterministic promo calendar (reconstruct rule from PH2)
PROMO_ALL = {"spring": (77, 30), "midyear": (174, 29), "fall": (242, 32), "yearend": (322, 45)}
PROMO_ODD = {"rural": (30, 30), "urban": (211, 34)}
is_promo = pd.Series(0, index=full_idx)
for d in full_idx:
    doy, yr = d.dayofyear, d.year
    for _, (s, dur) in PROMO_ALL.items():
        if s <= doy < s + dur: is_promo.loc[d] = 1
    if yr % 2 == 1:
        for _, (s, dur) in PROMO_ODD.items():
            if s <= doy < s + dur: is_promo.loc[d] = 1

# ---- Prophet Revenue (train <=2021) ----
mp = fit_prophet(df.loc[:TRAIN_END, "Clean_Revenue"].dropna(), "Revenue")
pp = predict_prophet(mp, str(df.index.min().date()), VAL_END)
train_prophet = pp.loc[:TRAIN_END, "prophet_pred"]
resid_train = df.loc[:TRAIN_END, "Revenue"].reindex(train_prophet.index) - train_prophet
full_resid = pd.Series(np.nan, index=full_idx); full_resid.update(resid_train)
full_trend = pp["prophet_trend"].reindex(full_idx)

# ---- build full candidate feature matrix ----
X = make_future_safe_features(full_resid, full_trend, full_idx)
X["sessions_lag365"]       = sessions.shift(365).reindex(full_idx)
X["overstock_pct_lag365"]  = overstock.shift(365).reindex(full_idx)
X["days_of_supply_lag365"] = days_supply.shift(365).reindex(full_idx)
X["fill_rate_lag365"]      = fill_rate.shift(365).reindex(full_idx)
X["is_promo_lag365"]       = is_promo.shift(365).reindex(full_idx)
X["is_promo_lag730"]       = is_promo.shift(730).reindex(full_idx)
X["is_odd_year"]           = (full_idx.year % 2).astype(int)

BASE = ["resid_lag365", "resid_lag364", "resid_lag366", "resid_roll28_lag365",
        "prophet_trend", "month", "dayofweek", "is_month_end", "weekofyear", "quarter", "dayofyear"]
CONFIGS = {
    "base":            [],
    "+sessions":       ["sessions_lag365"],
    "+overstock":      ["overstock_pct_lag365"],
    "+days_supply":    ["days_of_supply_lag365"],
    "+fill_rate":      ["fill_rate_lag365"],
    "+sess+overstock": ["sessions_lag365", "overstock_pct_lag365"],
    "+promo_lag365":   ["is_promo_lag365"],
    "+promo_lag730":   ["is_promo_lag730"],
    "+odd_year":       ["is_odd_year"],
    "ALL_inv+sess":    ["sessions_lag365", "overstock_pct_lag365", "days_of_supply_lag365", "fill_rate_lag365"],
}

val_prophet = pp.loc[VAL_START:VAL_END, "prophet_pred"]
actual_val = df.loc[VAL_START:VAL_END, "Revenue"]
y_resid = full_resid  # train residuals only non-NaN <=2021

print("\n" + "=" * 72)
print(f"{'config':18s} {'val MAE (M)':>12s} {'val R2':>9s} {'dMAE%':>8s}")
print("=" * 72)
base_mae = None
for name, extra in CONFIGS.items():
    feats = BASE + extra
    Xtr = X.loc[:TRAIN_END, feats].dropna()
    ytr = resid_train.reindex(Xtr.index).dropna()
    Xtr = Xtr.reindex(ytr.index)
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(Xtr, ytr)
    Xval = X.loc[VAL_START:VAL_END, feats]
    pred_resid = pd.Series(m.predict(Xval), index=Xval.index)
    pred = (val_prophet + pred_resid).reindex(actual_val.index)
    mae = mean_absolute_error(actual_val, pred)
    r2 = r2_score(actual_val, pred)
    if base_mae is None: base_mae = mae
    dmae = (mae - base_mae) / base_mae * 100
    flag = "  <-- giu" if (dmae <= -0.5) else ("  (bo)" if dmae > 0.5 else "")
    print(f"{name:18s} {mae/1e6:12.4f} {r2:9.4f} {dmae:+7.2f}%{flag}")
print("=" * 72)
print("Quy tac: giu neu dMAE <= -0.5%; bo neu dMAE > +0.5%; con lai trung tinh")
