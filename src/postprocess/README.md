# Postprocess Module

**Bước 4 — Tổng hợp kết quả và xuất submission.**

## Files

| File | Mô tả |
|------|-------|
| `blend.py` | Blend Prophet + LGBM, hậu xử lý, xuất file |
| `__init__.py` | Export: `blend_forecasts`, `postprocess` |

## Hàm chính

### `blend_forecasts(prophet_pred, lgbm_residual_pred)`
Cộng đơn giản:
```
Final_Forecast = Prophet_Prediction + LightGBM_Residual_Prediction
```
Trả về Series cùng index.

### `postprocess(revenue_pred, cogs_pred, sample_submission_path, output_path)`
Hậu xử lý và xuất `submission.csv`:

1. **Clip âm:** Revenue và COGS < 0 → set = 0
2. **Enforce COGS ≤ Revenue:** Nếu COGS > Revenue → COGS = Revenue × 0.85
3. **Round 2 chữ số thập phân**
4. **Kiểm tra số lượng dòng** phải khớp với `sample_submission.csv` (548 rows)
5. **Lưu ra file** theo đường dẫn chỉ định

## Ràng buộc tuân thủ
- Toàn bộ 548 dòng, giữ đúng thứ tự như `sample_submission.csv`
- Revenue ≥ 0, COGS ≥ 0
- COGS ≤ Revenue (business constraint)
- Không có NaN
