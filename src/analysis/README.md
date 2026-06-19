# Analysis Module

**Phần 2 — Trực quan hoá và Phân tích Dữ liệu (60 điểm).**

Sinh 14 biểu đồ theo 4 cấp độ phân tích cho `notebooks/Part2_EDA_Analysis.ipynb`.

## Files

| File | Cấp độ | Viz | Mô tả |
|------|--------|-----|-------|
| `descriptive.py` | Descriptive | 1–4 | Tổng quan doanh thu, margin heatmap, STL, category |
| `diagnostic.py` | Diagnostic | 5–8 | Denoising/stockout, promo intervention, web CCF, cointegration |
| `predictive.py` | Predictive | 9–11 | So sánh mô hình, forecast intervals, SHAP |
| `prescriptive.py` | Prescriptive | 12–14 | Safety stock, phân bổ marketing, early warning |
| `__init__.py` | — | — | Export 4 hàm `run_*` |

## 14 Biểu đồ

| # | File ảnh | Mô tả |
|---|----------|-------|
| 1 | `viz1_revenue_trend.png` | Doanh thu + rolling volatility band (2012–2022) |
| 2 | `viz2_profit_margin_heatmap.png` | Biên lợi nhuận gộp (%) heatmap theo tháng × năm |
| 3 | `viz3_stl_decomposition.png` | STL Decomposition: Trend / Seasonal / Residual |
| 4 | `viz4_revenue_by_category.png` | Doanh thu theo category (join products + order_items + orders) |
| 5 | `viz5_denoising_stockout.png` | Before/After denoising + highlight stockout periods |
| 6 | `viz6_promotion_intervention.png` | Event study: Revenue lift quanh ngày bắt đầu promo |
| 7 | `viz7_web_traffic_ccf.png` | Cross-correlation web sessions vs Revenue |
| 8 | `viz8_cointegration_anomaly.png` | Engle–Granger cointegration + spread anomaly detection |
| 9 | `viz9_model_comparison.png` | So sánh MAE/R²: Baseline vs Prophet Only vs Gridbreaker |
| 10 | `viz10_forecast_intervals.png` | Dự báo 18 tháng + khoảng tin cậy 95% |
| 11 | `viz11_shap_upgraded.png` | SHAP bar + beeswarm cho LGBM residual model |
| 12 | `viz12_safety_stock_tradeoff.png` | Trade-off: buffer % vs stockout risk % |
| 13 | `viz13_marketing_allocation.png` | Phân bổ marketing budget theo ROI score / quý |
| 14 | `viz14_early_warning_system.png` | Hệ thống cảnh báo sớm: web traffic YoY vs revenue YoY |

## Kết nối dữ liệu (Cross-table joins)

| Join | Mục đích |
|------|----------|
| `sales` ↔ `inventory` | Stockout imputation, denoising |
| `sales` ↔ `promotions` | Intervention analysis |
| `sales` ↔ `web_traffic` | Leading indicator CCF |
| `products` ↔ `order_items` ↔ `orders` | Phân tích doanh thu theo category |
| `Revenue` ↔ `COGS` (nội bộ) | Cointegration & margin analysis |
| Forecast errors ↔ `inventory` | Safety stock trade-off |

## Cách chạy
```bash
python generate_analysis.py
# Output: outputs/part2/ (14 file PNG)
```
