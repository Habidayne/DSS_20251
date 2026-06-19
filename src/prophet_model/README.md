# Prophet Model Module

**Bước 2 — Xây dựng baseline cấu trúc: Trend + Seasonality.**

## Files

| File | Mô tả |
|------|-------|
| `train_prophet.py` | Huấn luyện và dự báo với Prophet |
| `__init__.py` | Export: `fit_prophet`, `predict_prophet` |

## Hàm chính

### `fit_prophet(series, target_name)`
Huấn luyện Prophet trên `Clean_Revenue` hoặc `Clean_COGS`.

Cấu hình:
- **Yearly seasonality:** Fourier series (học chu kỳ năm)
- **Weekly seasonality:** Fourier series (học chu kỳ tuần)
- **Changepoint prior scale = 0.15** (linh hoạt hơn default 0.05 để bắt trend shifts)
- **Custom holidays (inject thủ công):**
  - `end_of_month`: 3 ngày cuối tháng (nhu cầu tăng đột biến)
  - `start_of_month_recovery`: 3 ngày đầu tháng kế tiếp
  - `tet_nguyen_dan`: Lễ Tết Nguyên Đán (chu kỳ năm)
  - `black_friday`: Black Friday (chu kỳ năm)

### `predict_prophet(model, start_date, end_date)`
Dự báo trên khoảng thời gian [start_date, end_date], trả về DataFrame với:
- `prophet_pred`: Giá trị dự báo
- `prophet_trend`: Component xu hướng (dùng làm feature cho LGBM)
- `prophet_lower`: Giới hạn dưới 95% CI
- `prophet_upper`: Giới hạn trên 95% CI

## Input/Output
- **Input:** `Clean_Revenue` hoặc `Clean_COGS` series (từ Denoising)
- **Output:** `prophet_pred` + `prophet_trend` cho cả train và test period
