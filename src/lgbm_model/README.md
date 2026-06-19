# LightGBM Residual Module

**Bước 3 — Học phần dư phi tuyến: Residual = Actual − Prophet_Prediction.**

## Files

| File | Mô tả |
|------|-------|
| `train_lgbm.py` | Huấn luyện và dự báo residual với LightGBM |
| `__init__.py` | Export: `fit_lgbm_residual`, `predict_lgbm_residual`, `make_future_safe_features`, `FEAT_COLS` |

## Features (100% future-safe — KHÔNG data leakage)

| Feature | Mô tả | Tại sao an toàn? |
|---------|-------|-----------------|
| `resid_lag365` | Residual cùng ngày năm ngoái | Shift 365 ngày |
| `resid_lag364` | Residual cách 364 ngày | Shift 364 ngày |
| `resid_lag366` | Residual cách 366 ngày | Shift 366 ngày |
| `resid_roll28_lag365` | Rolling mean 28 ngày của lag365 | Luôn có sẵn |
| `prophet_trend` | Component xu hướng từ Prophet | Deterministic |
| `month` | Tháng trong năm (1–12) | Calendar |
| `dayofweek` | Thứ trong tuần (0=Mon) | Calendar |
| `is_month_end` | Có là ngày cuối tháng | Calendar |
| `weekofyear` | Tuần trong năm | Calendar |
| `quarter` | Quý (1–4) | Calendar |
| `dayofyear` | Ngày thứ mấy trong năm | Calendar |

**Tổng cộng: 11 features.**

## Cấu hình mô hình (LGBMRegressor)
- `n_estimators = 800`
- `learning_rate = 0.03`
- `num_leaves = 63`
- `max_depth = 8`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `reg_alpha = 0.1`, `reg_lambda = 1.0`
- `random_state = 42`

## Hàm chính

### `make_future_safe_features(residuals, prophet_trend, index)`
Tạo feature matrix cho bất kỳ khoảng thời gian nào (train hoặc test).

### `fit_lgbm_residual(X_train, y_train, target_name)`
Huấn luyện LightGBM trên residuals. Tự động drop NaN (lag_365 cần ≥ 1 năm data).

### `predict_lgbm_residual(model, X)`
Dự đoán residual. Fill NaN bằng 0 trước khi predict.

## Input/Output
- **Input:** Residuals từ Bước 2 (`Actual − Prophet_pred`)
- **Output:** `lgbm_residual_pred` series cùng index với test period
