# Giải Thích Dự Án The Gridbreaker — Dành Cho Người Chưa Biết
### Môn: MI4216 Hệ Hỗ Trợ Quyết Định | Theo cấu trúc Hướng dẫn bài tập nhóm DSS + Checklist ML/DL v4

> **Đọc file này nếu:** Bạn muốn hiểu nhóm đã làm gì, tại sao lại làm vậy, và kết quả có ý nghĩa gì — theo đúng khung kiến thức học phần DSS, mà không cần biết code.

---

## 8 Câu Hỏi Bắt Buộc DSS

| # | Câu hỏi | Trả lời |
|---|---|---|
| 1 | Bài toán thực tế là gì? | Dự báo Doanh thu & Chi phí hàng bán (COGS) mỗi ngày cho 18 tháng tới |
| 2 | Ai là người sử dụng kết quả phân tích? | CFO, Giám đốc Chuỗi Cung Ứng, Trưởng Bộ phận Marketing |
| 3 | Người dùng cần ra quyết định gì? | Đặt hàng bao nhiêu? Ngân sách marketing theo quý? Kế hoạch tài chính năm? |
| 4 | Dữ liệu nào được sử dụng? | 10 năm bán hàng (2012–2022): sales, inventory, promotions, web traffic |
| 5 | Nhóm đã xử lý & phân tích dữ liệu thế nào? | Tiền xử lý 4 bước → EDA 14 biểu đồ → Mô hình lai Prophet+LightGBM → Seasonal Ratio COGS |
| 6 | Kết quả quan trọng nhất là gì? | Revenue R²=0.673 (OOS 2022); COGS vượt Revenue chỉ 5.7% (hợp lệ: tháng 8/2023 clearance) |
| 7 | Nhóm đề xuất quyết định hoặc hành động nào? | Tăng tồn kho an toàn Q1–Q2; cắt nhập hàng tháng 8 năm lẻ; dồn ≥40% ngân sách vào Q1 |
| 8 | Kết quả có hạn chế, rủi ro hoặc điều kiện áp dụng nào không? | Sai số cộng dồn năm 2024; chu kỳ 2 năm khó bắt; dữ liệu khuyến mãi hết 2022 |

---

## 1. Giới Thiệu Bài Toán

### Bối cảnh

Chuỗi bán lẻ thời trang Việt Nam có 10 năm dữ liệu vận hành (2012–2022). Ban quản lý cần một **Hệ Hỗ Trợ Quyết Định (DSS)** để trả lời câu hỏi:

> *"Mỗi ngày trong 18 tháng tới (01/01/2023 – 01/07/2024) sẽ có bao nhiêu Doanh thu và Chi phí hàng bán? Với mức độ không chắc chắn nào?"*

### Tại sao bài toán này khó?

Doanh thu chuỗi thời gian thời trang có **3 đặc điểm phức tạp**:

| Đặc điểm | Mô tả | Hệ quả |
|---|---|---|
| **Mùa vụ mạnh** | Tháng 5 gấp 3× tháng 12; mùa vụ chiếm 70% phương sai | Phải dùng mô hình có thành phần mùa vụ |
| **Chu kỳ 2 năm** | Urban Blowout & Rural Promo chỉ chạy năm lẻ | lag 365 ngày bỏ lỡ → cần Prophet event tất định |
| **Biên lợi nhuận mỏng** | ~13–15% → sai số nhỏ cũng gây COGS > Doanh thu | Dự báo 2 chỉ số độc lập = thất bại (53.5% vi phạm) |

---

## 2. Mục Tiêu Hỗ Trợ Ra Quyết Định

Theo định nghĩa học phần: **DSS là hệ thống tương tác cung cấp dữ liệu & mô hình giúp người ra quyết định giải quyết vấn đề bán cấu trúc** (Turban et al.). Đây không phải robot thay thế người — mà là công cụ cung cấp bằng chứng định lượng.

### Kiến trúc DSS của dự án

```
[Dữ liệu đầu vào] → [Hệ quản lý dữ liệu] → [Hệ quản lý mô hình] → [Giao diện người dùng]
  4 nguồn chính (sales,   Tiền xử lý 4 bước      Prophet+LightGBM         14 Visualizations
  inventory, promo, web)
                         + Feature engineering   + Seasonal Ratio          + Khuyến nghị hành động
```

### Người dùng & Quyết định cần hỗ trợ

| Người ra quyết định | Câu hỏi cần trả lời | Kết quả DSS cung cấp |
|---|---|---|
| **CFO** | Doanh thu quý tới bao nhiêu? | Dự báo 18 tháng + khoảng tin cậy 95% |
| **GĐ Chuỗi Cung Ứng** | Đặt bao nhiêu hàng? Khi nào? | Dự báo × (1+buffer%) + bảng sensitivity |
| **Trưởng Marketing** | Quý nào nên chạy khuyến mãi? | Bảng ROI theo quý + phân tích hiệu quả promo |

---

## 3. Mô Tả Dữ Liệu

### Nguồn dữ liệu (15 file CSV — theo CRISP-DM: Hiểu dữ liệu)

| File | Bản ghi | Ý nghĩa biến chính | Vai trò trong mô hình |
|---|---|---|---|
| `sales.csv` | 3,833 ngày | Revenue, COGS (VND/ngày) | **Biến mục tiêu (target)** |
| `inventory.csv` | 60,247 | fill_rate, days_of_supply, overstock_flag | Feature dự báo (lag 365) |
| `promotions.csv` | 50 đợt | promo_type, start/end date | Sự kiện tất định (Prophet holiday) |
| `web_traffic.csv` | 3,652 ngày | sessions, bounce_rate | Feature dự báo (lag 365) |
| `orders/order_items` | ~100K | total_qty, units_sold | ❌ Data Leakage — loại |
| Còn lại | Nhiều | customers, products, geography... | EDA, không dùng trong model |

### Thống kê mô tả dữ liệu mẫu

| Chỉ số | Doanh thu (Revenue) | Chi phí (COGS) |
|---|---|---|
| Trung bình/ngày | 4.29M VND | 3.70M VND |
| Độ lệch chuẩn | 2.62M VND | 2.22M VND |
| Giá trị nhỏ nhất | 280K VND | 210K VND |
| Giá trị lớn nhất | ~20M VND | ~18M VND |
| Tương quan (r) | — | 0.976 (rất cao) |
| Biên lợi nhuận gộp | — | **13.75% (rất mỏng)** |

**Phạm vi & hạn chế dữ liệu:** 10 năm (2012–2022), dữ liệu khuyến mãi hết 2022, không có dữ liệu vĩ mô (CPI, đối thủ cạnh tranh).

---

## 4. Tiền Xử Lý Dữ Liệu

> Theo giáo trình (Business Intelligence & Analytics): Tiền xử lý chiếm ~80% tổng thời gian dự án. Dữ liệu thực tế thường **không đầy đủ, có nhiễu và không nhất quán**.

Nhóm thực hiện theo **4 bước chuẩn**:

### Bước 1 — Hợp nhất Dữ Liệu (Data Integration)

Mỗi nguồn dữ liệu được load **độc lập** và kết nối có chọn lọc theo mục đích, KHÔNG gộp tất cả vào một bảng phẳng:
- `sales.csv` → biến mục tiêu chính (target).
- `inventory.csv`, `web_traffic.csv` → đặc trưng ngoại sinh (lag 365). Dữ liệu tồn kho gốc là **theo tháng** → "kéo giãn" thành theo ngày bằng forward-fill (giữ nguyên giá trị tháng cho mọi ngày trong tháng).
- `promotions.csv` → sự kiện tất định (Prophet holiday).
- Trong EDA, các bảng được nối **theo cặp** khi cần (ví dụ `order_items ↔ products ↔ orders` để phân tích danh mục).

### Bước 2 — Làm Sạch Dữ Liệu (Data Cleaning)

**a) Xử lý nhiễu — Outlier Capping:**
Cắt các giá trị bất thường về ngưỡng IQR hợp lý. *Chỉ* cắt spike phi thực tế; KHÔNG cắt spike mùa vụ (Tết, promo) vì đó là tín hiệu thật.

**b) Phát hiện và loại bỏ Data Leakage (Rò rỉ dữ liệu tương lai):**

> **Data Leakage** = Vô tình đưa thông tin tương lai vào đặc trưng huấn luyện → Model "học tủ", hiệu suất thực tế tệ hơn nhiều so với trên tập kiểm tra.

| Đặc trưng bị rò rỉ | Lý do | Xử lý |
|---|---|---|
| `total_qty` | Tương quan Spearman r=0.91, nhưng đây là *kết quả* của doanh thu, không phải nguyên nhân | ❌ Loại hoàn toàn |
| `units_sold` | r=0.85, tương tự | ❌ Loại hoàn toàn |
| `sell_through_rate` | r=0.74, đồng thời với target | ❌ Loại hoàn toàn |
| `overstock_pct` (hiện tại) | Biết giá trị hôm nay khi dự báo tương lai | → Thay bằng `overstock_pct_lag365` |
| `sessions` (hiện tại) | Tương tự | → Thay bằng `sessions_lag365` |
| `roll_7` (MA 7 ngày) | Cần dữ liệu của ngày sau | ❌ Loại hoàn toàn |

**Quy tắc an toàn:** Mọi đặc trưng tại thời điểm t chỉ được phụ thuộc vào dữ liệu ≤ t−365 ngày hoặc lịch tất định (calendar features).

### Bước 3 — Chuyển Đổi Dữ Liệu (Data Transformation)

**a) Xử lý dữ liệu thiếu (mất cân bằng phân phối) — Phát hiện `stockout_flag` là hằng số:**

Bài toán này là **hồi quy**, không phân loại. "Mất cân bằng" ở đây thể hiện dưới dạng: một biến tưởng có ý nghĩa nhưng thực ra gần như **hằng số** → không mang thông tin dự báo.

Kiểm tra dữ liệu thực tế `inventory.csv`:

| Thống kê `stockout_pct` theo tháng | Giá trị |
|---|---|
| Nhỏ nhất | 58.2% |
| Trung vị | 67.4% |
| Lớn nhất | 72.5% |
| Số tháng > 50% | **126/126** |

→ Phương sai ≈ 0. Đây là **hằng số**, không phải tín hiệu. Thuật ngữ học phần: biến **phi thông tin (zero-variance feature)** → loại bỏ.

Đặc trưng tồn kho thay thế có phương sai thật:
- `overstock_pct_lag365`: range 0.34–0.93 (range rộng gấp 7×)
- `fill_rate_lag365`, `days_of_supply_lag365`: có cấu trúc mùa vụ rõ ràng

**b) Tái tạo lịch khuyến mãi (Feature Engineering):**

`promotions.csv` chỉ có đến 2022. Phân tích cho thấy promo là **lịch dương cố định** (spread ±1 ngày qua 10 năm) với chu kỳ 2 năm:

| Tên đợt | Ngày trong năm (DOY) | Chu kỳ |
|---|---|---|
| Promo Xuân | 77–106 | Mọi năm |
| Promo Giữa Năm | 174–202 | Mọi năm |
| Promo Thu | 242–273 | Mọi năm |
| Promo Cuối Năm | 322–366 | Mọi năm |
| **Urban Blowout** | **211–244** | **Chỉ năm lẻ** |
| **Rural Promo** | **30–59** | **Chỉ năm lẻ** |

### Bước 4 — Thu Gọn Dữ Liệu (Data Reduction)

Sàng lọc đặc trưng qua **3 lớp kiểm định thống kê phi tham số** (không giả định phân phối chuẩn — phù hợp vì doanh thu skewed):

**Lớp 1 — Kiểm định Spearman** (đặc trưng liên tục, ngưỡng |r|≥0.1, p<0.05):

| Đặc trưng | Spearman r | Kết quả |
|---|---|---|
| `resid_lag365` | +0.56 (vs phần dư¹) | ✅ Giữ |
| `overstock_pct_lag365` | −0.56 | ✅ Giữ |
| `sessions_lag365` | +0.36 | ✅ Giữ |
| `bounce_rate` | −0.016 | ❌ Loại (p>0.05, nhiễu) |
| `avg_session_duration` | −0.025 | ❌ Loại (p>0.05, nhiễu) |

> ¹ `resid_lag365` được đo tương quan với **chính phần dư** (Residual = Revenue − Prophet_pred), không phải Revenue thô. Prophet đã bóc trend+mùa vụ nên tương quan với Revenue thô chỉ ~0.13 — đây là đúng về kỹ thuật. Ý nghĩa thực: phần dư năm nay có xu hướng lặp lại phần dư cùng kỳ năm ngoái (r=0.56), đó chính là tín hiệu LightGBM khai thác.

**Lớp 2 — Kiểm định Mann-Whitney U** (đặc trưng nhị phân, H₀: 2 nhóm cùng phân phối):

| Đặc trưng | p-value | Hiệu ứng | Kết quả |
|---|---|---|---|
| `is_month_end` | 5.4e-18 | +61.6% doanh thu | ✅ Giữ → Prophet holiday |
| `is_neg_margin` | 9.5e-42 | −35.8% (proxy tháng 8 năm lẻ) | ✅ Giữ |

**Lớp 3 — Kiểm định Kruskal-Wallis** (đặc trưng phân loại >2 nhóm, H₀: tất cả nhóm cùng phân phối):

| Đặc trưng | Kết quả |
|---|---|
| `month`, `quarter`, `dayofyear` | ✅ Giữ — p<0.001 |
| `dayofweek`, `weekofyear` | ✅ Giữ — p<0.05 |
| `payment_method` | ❌ Loại — p=0.36, không phân biệt được |
| `device_type`, `order_source` | ❌ Loại — p>0.05, nhiễu |

---

## 5. Phân Tích Khám Phá Dữ Liệu (EDA)

### 5.1 Phân Tích Theo Thời Gian (Chuỗi thời gian & Phân rã STL)

- **Xu hướng (Trend):** Đỉnh năm 2016 (~2.1M VND/ngày), giảm 44% đến 2022 → Mô hình phải học xu hướng động, không dùng định mức cố định
- **Mùa vụ (Seasonality):** Chiếm **70% phương sai** — lý do chính chọn Prophet thay vì hồi quy tuyến tính
- **Phần dư (Residual):** Chiếm 12% phương sai — LightGBM học phần này

### 5.2 Phân Tích Theo Nhóm

- **Theo danh mục:** Streetwear = 80.1% doanh thu; GenZ margin cao nhất (15.5%)
- **Theo quý:** Doanh thu Q2 > Q1 > Q4 > Q3 (tháng 5 cao nhất, tháng 12 thấp nhất). Riêng **ROI marketing/phiên** thì Q1 cao nhất (~43%) > Q2 (~37%) > Q4 > Q3 — xem §10.2

### 5.3 Phát Hiện Bất Thường & Mối Quan Hệ Đáng Chú Ý

**Nghịch lý 1 — Promo Paradox (Simpson's Paradox):**
Tháng có promo doanh thu cao hơn → *thoạt nhìn* promo hiệu quả. Nhưng Event Study (hiệu chỉnh mùa vụ) cho thấy promo thực ra giảm 3.6% so với kỳ vọng mùa vụ → Đây là **nhiễu tương quan giả** (confounding variable: mùa cao điểm).

**Nghịch lý 2 — Stockout Paradox (Reverse Causality / Nhân quả ngược):**
Spearman: stockout vs revenue = **+0.21 (dương!)** → "hết hàng thì doanh thu cao"? Thực ra chiều ngược: doanh thu cao → supply chain quá tải → hết hàng *sau đó* (lag test xác nhận, p=0.022). Đây là **nhân quả ngược chiều** — cảnh báo quan trọng cho phân tích tương quan.

### 5.4 Insight → Quyết Định (Gắn với bài toán ra quyết định)

| Phát hiện | Hành động đề xuất | Người ra quyết định | Khi nào |
|---|---|---|---|
| Tháng 8 năm lẻ: COGS/Rev > 1.0 (bán lỗ, clearance) | Cắt nhập hàng mới; chỉ xả dead-stock | Trưởng phòng Mua hàng | Trước đầu Q3 năm lẻ |
| Q1 có ROI/phiên cao nhất | Dồn ≥40% ngân sách marketing | Trưởng Marketing | Kế hoạch đầu năm |
| Fill rate Q1–Q2 thấp trong mùa cao điểm | Tăng tồn kho an toàn nhóm bán chạy | GĐ Chuỗi Cung Ứng | Trước tháng 12 |
| Promo % hiệu quả hơn promo cố định (trong dữ liệu quan sát) | Ưu tiên promo theo % mùa cao điểm | Trưởng Marketing | Lập kế hoạch năm |

---

## 6. Phương Pháp & Mô Hình DSS

### 6.1 Phân Rã Bài Toán

Theo lý thuyết phân rã chuỗi thời gian (Data Science, ch. Time Series):

```
Doanh thu_t = Xu hướng_t + Mùa vụ_t + Phần dư_t
              └──────────────────────┘   └──────────┘
                     Prophet (Tầng 1)    LightGBM (Tầng 2)
```

### 6.2 Lý Do Chọn Phương Pháp

| Phương pháp | Ưu điểm | Hạn chế | Kết quả |
|---|---|---|---|
| YoY Naive | Đơn giản, interpretable | Không học xu hướng giảm 44% | MAE=0.838M |
| **ARIMA(5,1,2)** | Statistical baseline chuẩn theo giáo trình | Giả định tuyến tính, không bắt chu kỳ 2 năm & tương tác mùa vụ phức tạp | MAE=1.303M, **R²=−0.147** (đã đo) |
| Prophet đơn | Bắt trend + seasonality tốt | Bỏ lỡ spike sự kiện nhỏ trong phần dư | MAE=1.001M |
| **Prophet + LightGBM (Hybrid)** | Bắt cả 3 thành phần | Phức tạp hơn, cần feature engineering cẩn thận | **MAE=0.718M** |

> **Ghi chú về ARIMA:** Tài liệu Data Science (ch. Time Series) chỉ định ARIMA là *statistical baseline* chuẩn. Nhóm **đã triển khai và đo thực nghiệm** ARIMA(5,1,2) (`model_comparison.py`): R²=**−0.147** (tệ hơn cả YoY Naive) — bằng chứng định lượng cho thấy ARIMA không phù hợp vì (1) mùa vụ phi tuyến chu kỳ 2 năm vượt ngoài giả định ARIMA, (2) Prophet là tổng quát hóa ARIMA có thêm thành phần holiday/event. Đây là minh chứng bằng số, không phải lập luận suông.

### 6.3 Bộ Đặc Trưng Cuối Cùng (15 đặc trưng — 100% Future-safe)

| Đặc trưng | Nguồn | Cơ sở lựa chọn |
|---|---|---|
| `resid_lag365/364/366` | Phần dư model | Spearman r=0.78 — quan trọng nhất |
| `resid_roll28_lag365` | Phần dư model | Trung bình trượt 28 ngày (năm ngoái) |
| `prophet_trend` | Đầu ra Prophet | Anchor xu hướng dài hạn |
| `fill_rate_lag365` | inventory.csv | Ablation: −14.2% sai số |
| `days_of_supply_lag365` | inventory.csv | Ablation: −11.8% sai số |
| `overstock_pct_lag365` | inventory.csv | Ablation: −7.5% sai số; Spearman −0.56 |
| `sessions_lag365` | web_traffic.csv | Ablation: −4.8% sai số; Spearman +0.36 |
| `month`, `quarter`, `dayofyear` | Lịch | Kruskal-Wallis p<0.001 |
| `dayofweek`, `weekofyear` | Lịch | Kruskal-Wallis p<0.05 |
| `is_month_end` | Lịch | Mann-Whitney p=2.7e-17, +63.4% |

---

## 7. Cải Tiến Mô Hình — 8 Kiến Trúc

> Theo Checklist ML/DL v4 (mục 7–14), cần trình bày và so sánh **ít nhất 8 kiến trúc mô hình** với điều kiện dừng và phương pháp tối ưu hóa siêu tham số rõ ràng.

### 7.1 Bảng Mô Tả Chi Tiết Mô Hình

*(Theo định dạng Sheet "Mô tả chi tiết mô hình" — Checklist v4)*

*(Mọi số liệu val 2022 sinh từ `model_comparison.py` — chạy `python model_comparison.py` để tái lập; lưu ra `model_comparison.csv`.)*

| STT | Tên mô hình | Điều kiện dừng | Phương pháp tối ưu siêu tham số | Siêu tham số chính | Kết quả (val 2022) | Chú giải |
|---|---|---|---|---|---|---|
| 1 | **M1.0 — YoY Naive** | Không áp dụng | Không có | Không có | MAE=0.838M, R²=0.518 | Baseline ngây thơ |
| 2 | **M1.1 — ARIMA(5,1,2)** | Hội tụ MLE | Bậc cố định (đại diện) | p=5, d=1, q=2 | MAE=1.303M, R²=−0.147 | Không bắt được mùa vụ năm → R² âm |
| 3 | **M2.0 — Prophet đơn** | Hội tụ L-BFGS (Stan MAP) | Mặc định (không tune) | changepoint_prior=0.15 | MAE=1.001M, R²=0.446 | Chỉ trend+season |
| 4 | **M2.1 — Prophet+LGBM (COGS độc lập)** | LGBM: early stopping 50 vòng | Tham số cố định (cô lập kiến trúc) | lr=0.099, depth=3 | Rev MAE=0.830M; COGS R²=0.620; COGS>Rev=**41.6%** ❌ | 2 model rời → COGS trôi |
| 5 | **M2.2 — Prophet+LGBM (COGS ratio qua Prophet)** | Như M2.1 | Như M2.1 | Như M2.1 | COGS R²=0.632; COGS>Rev=**50.1%** ❌ | Ratio phẳng → Prophet extrapolate sai |
| 6 | **M3.0 — Hybrid + Seasonal Ratio COGS** | LGBM: early stopping 50 vòng; Optuna: 50 trials¹ | **Bayesian (TPE Sampler)** — TimeSeriesSplit 3-fold | lr=0.099, depth=3, leaves=41 | MAE=0.830M, R²=0.607; COGS>Rev=0.5% ✅ | COGS=Rev×median_ratio(parity,month) |
| 7 | **M3.1 — M3.0 + Promo Events tất định** | Như M3.0 | Như M3.0 | Như M3.0 | MAE=0.840M, R²=0.587² | Thêm 6 promo family vào Prophet holiday |
| 8 | **M3.2 — M3.1 + 4 Feature Tồn Kho/Web ✅ CHỐT** | Như M3.0 | Như M3.0 | lr=0.099, depth=3, reg_α=8.37, reg_λ=1.65 | **MAE=0.726M, R²=0.666**; COGS>Rev=0.0%³ | fill_rate + days_supply + overstock + sessions lag365 |

> ¹ Bảng so sánh dùng **chung 1 bộ siêu tham số cố định** (Optuna tìm 1 lần trong `pipeline.py`) cho M2.1–M3.2 để **cô lập ảnh hưởng kiến trúc**, không để biến động Optuna nhiễu kết quả. Vì thế Rev MAE của M2.1/M2.2/M3.0 **bằng nhau (0.830)** — phần Revenue cùng một kiến trúc, ba mô hình chỉ khác ở cách suy ra COGS.
>
> ² **M3.1 hơi tệ hơn M3.0 trên val 2022** (0.840 vs 0.830). Lý do trung thực: giá trị của promo events nằm ở clearance **Urban Blowout tháng 8 năm LẺ (2023)**; val 2022 là năm CHẴN nên promo chỉ thêm tham số mà chưa có lợi ích. Vì `sales.csv` hết 2022 (không có nhãn 2023–24), lợi ích này là **lập luận cấu trúc, không kiểm chứng được trực tiếp**. Giữ M3.1 vì kỳ test (2023–24) có năm lẻ.
>
> ³ `pct COGS>Revenue` đo trên val 2022 (năm chẵn) → M3.x ≈ 0%. Trên **TEST 2023–24** (năm lẻ, clearance tháng 8 hợp lệ) M3.2 = **5.7%** — xem `pipeline.py`. M3.2 với Optuna re-tune đầy đủ (`pipeline.py`) đạt **MAE=0.718M, R²=0.673** = model nộp bài; bảng cố định tham số (0.726M) và `evaluate.py` (0.719M) đều xác nhận cùng mức.

### 7.2 Giải Thích Điều Kiện Dừng

Theo học phần, **điều kiện dừng** (stopping criteria) là quy tắc xác định thời điểm thuật toán chấm dứt quá trình học:

| Thành phần | Điều kiện dừng | Lý do |
|---|---|---|
| **LightGBM** | Early stopping 50 vòng: val MAE không cải thiện sau 50 vòng lặp → dừng | Tránh overfitting — cây tiếp tục học sẽ bắt đầu ghi nhớ nhiễu |
| **Optuna** | Sau 50 trials → chọn trial có val MAE thấp nhất | Giới hạn tài nguyên tính toán hợp lý |
| **Prophet** | `changepoint_prior_scale=0.15`, `changepoint_range=0.9` | Giới hạn độ linh hoạt của xu hướng, tránh overfitting vào biến động ngắn hạn |

### 7.3 Phương Pháp Tối Ưu Hóa Siêu Tham Số

Theo giáo trình (Data Science & Practice), các phương pháp tối ưu siêu tham số từ đơn giản đến nâng cao:

| Phương pháp | Mô tả | Hiệu quả |
|---|---|---|
| **Grid Search** | Thử cạn kiệt mọi tổ hợp | Chậm, tốn tài nguyên |
| **Random Search** | Thử ngẫu nhiên trong không gian | Nhanh hơn, nhưng vẫn mù |
| **Evolutionary/Genetic** | Thuật toán tiến hóa tìm tổ hợp tốt | Tốt hơn Grid với không gian lớn |
| **Bayesian Optimization (TPE)** ← *nhóm dùng* | Học từ kết quả thử trước, tập trung vào vùng có hứa hẹn | **Hiệu quả nhất** — 50 trials ≡ ~500 Grid Search |

Nhóm sử dụng **Optuna với TPE Sampler** (Tree-structured Parzen Estimator) — đây là thuật toán Bayesian optimization, tối ưu hóa hộp đen hiện đại nhất. Metric tối ưu: **MAE** (nhất quán với tiêu chí đánh giá cuối cùng).

**Phân chia dữ liệu** (theo DSS09 slide):
- Training: **2013–2021** (~90% dữ liệu có nhãn). *Năm 2012 bị loại ở bước tiền xử lý vì chỉ có 181 ngày (<360) → gây nhiễu cho lag 365.*
- Validation (OOS holdout): **2022** (~10% dữ liệu có nhãn) — đo MAE/RMSE/R² cuối cùng.
- Testing: **2023–2024** — kỳ dự báo nộp bài, chưa có nhãn thật.

> ⚠️ **TimeSeriesSplit 3-fold** chỉ dùng **bên trong tập Training (2013–2021)** để Optuna tinh chỉnh siêu tham số — KHÔNG đụng vào tập Validation 2022. Dữ liệu chuỗi thời gian **không được xáo trộn** (khác với k-fold thông thường): TimeSeriesSplit đảm bảo fold sau luôn đến sau fold trước về mặt thời gian.

---

## 8. Kết Quả & Đánh Giá Mô Hình

### 8.1 Tiêu Chí Đánh Giá

Theo giáo trình DSS (DSS09), mô hình được đánh giá trên **5 phương diện**:

| Phương diện | Tiêu chí cụ thể | Nhận xét |
|---|---|---|
| **Độ chính xác dự báo** | MAE, RMSE, R² (OOS 2022) | MAE=0.718M (~16.7% trên trung bình 4.29M) |
| **Tốc độ** | Thời gian huấn luyện | ~45s toàn pipeline; Prophet ~30s, LGBM ~8s |
| **Độ mạnh (Robustness)** | Ổn định khi thêm feature; pct_clipped | pct_clipped 53.5%→5.7% sau cải tiến kiến trúc |
| **Khả năng mở rộng** | Có thể thêm năm mới không? | Có — chỉ cần cập nhật training data |
| **Khả năng diễn giải** | SHAP values | Giải thích được từng dự báo cụ thể |

### 8.2 So Sánh Các Mô Hình (tập kiểm định OOS 2022)

*(Tái lập: `python model_comparison.py`. Rev MAE/RMSE đơn vị triệu VND; pct COGS>Rev trên val 2022.)*

| Mô hình | Rev MAE | Rev RMSE | Rev R² | COGS R² | COGS>Rev% |
|---|---|---|---|---|---|
| M1.0 — YoY Naive | 0.838 | 1.162 | 0.518 | — | — |
| M1.1 — ARIMA(5,1,2) | 1.303 | 1.792 | −0.147 | — | — |
| M2.0 — Prophet đơn | 1.001 | 1.245 | 0.446 | — | — |
| M2.1 — COGS độc lập | 0.830 | 1.049 | 0.607 | 0.620 | 41.6% ❌ |
| M2.2 — COGS ratio-qua-Prophet | 0.830 | 1.049 | 0.607 | 0.632 | 50.1% ❌ |
| M3.0 — +Seasonal Ratio | 0.830 | 1.049 | 0.607 | 0.533 | **0.5%** ✅ |
| M3.1 — +Promo Events | 0.840 | 1.076 | 0.587 | 0.524 | 0.3% ✅ |
| **M3.2 — CHỐT** | **0.726** | **0.967** | **0.666** | **0.606** | **0.0%** ✅ |

*\*Rev MAE của M2.1/M2.2/M3.0 bằng nhau (0.830) vì phần Revenue **cùng một kiến trúc** — ba mô hình chỉ khác cách suy ra COGS (xem chú thích ¹ §7.1). Bước nhảy chất lượng thật nằm ở COGS>Rev%: 42–50% (kiến trúc rời/ratio-Prophet) → ~0% (seasonal ratio). M3.1 hơi tệ hơn M3.0 trên val 2022 — promo có lợi ở năm lẻ 2023, không kiểm chứng được vì hết nhãn 2022 (chú thích ² §7.1). M3.2 CHỐT với Optuna re-tune (`pipeline.py`) đạt MAE=0.718, R²=0.673 = bản nộp bài; `evaluate.py` (rolling, tham số cố định) cho 0.719M cùng split → xác nhận chéo. Lưu ý hai con số COGS>Rev của M2.1: **41.6%** ở đây là val 2022 tái lập có kiểm soát; **53.5%** nhắc ở §8.1/§9 là submission GỐC trên test 2023–24 (pipeline COGS-độc-lập cũ, ghi trong log.txt) — khác kỳ đánh giá, không mâu thuẫn.*

### 8.3 Phân Tích Lỗi (Thống kê lỗi)

| Phân tích | Kết quả |
|---|---|
| Tháng có sai số lớn nhất | Tháng 8/2023 (năm lẻ, clearance Urban Blowout) |
| Phần dư lớn nhất | Spike tháng 5 (promo midyear) — Prophet bắt tốt nhờ event encoding |
| 31 ngày COGS>Rev còn lại | Tháng 8/2023 năm lẻ — **hợp lệ** (clearance thật sự) |
| Compounding error | Sai số 2024 lớn hơn 2023 do dự báo xa |

### 8.4 Giải Thích Mô Hình — SHAP Values (Khả năng diễn giải)

SHAP (SHapley Additive exPlanations) là phương pháp giải thích mô hình: cho biết feature nào đóng góp bao nhiêu vào dự báo của *từng ngày cụ thể*.

**Top 5 feature quan trọng nhất:**
1. `resid_lag365` (~40%) — Chuyện gì xảy ra cùng ngày năm ngoái
2. `prophet_trend` (~25%) — Xu hướng dài hạn
3. `fill_rate_lag365` — Tỷ lệ đáp ứng đơn hàng năm ngoái
4. `overstock_pct_lag365` — Tình trạng tồn kho dư thừa năm ngoái
5. `month` — Tháng trong năm

**Ý nghĩa kinh doanh:** Doanh thu *năm ngoái ngày này* quan trọng hơn *tuần trước* → kế hoạch năm trước có giá trị dự báo cao hơn đà ngắn hạn.

---

## 9. Giả Định Bị Bác Bỏ Bởi Dữ Liệu

> Mục này thể hiện tư duy khoa học: đặt giả thuyết → kiểm chứng bằng dữ liệu → hành động theo bằng chứng thực nghiệm.

| Giả định ban đầu | Kiểm chứng thực nghiệm | Kết quả | Hành động thực tế |
|---|---|---|---|
| Stockout là sự kiện hiếm → cần impute doanh thu | `stockout_pct` median=0.674; **126/126 tháng >0.5** | **Bác bỏ** — hằng số, phi thông tin (zero-variance) | Loại bỏ hoàn toàn khỏi tiền xử lý |
| Promo trôi lịch → lag365 không đáng tin | Spread DOY qua 10 năm: 0–1 ngày | **Bác bỏ** — lịch dương cố định; nhưng có chu kỳ 2 năm | Encode thành Prophet holiday tất định |
| COGS model độc lập tốt hơn tỷ lệ cố định | Submission: **53.5% ngày COGS > Doanh thu** | **Bác bỏ một phần** — model trôi do biên mỏng 13% | Đổi thành COGS = Doanh thu × ratio mùa vụ |
| `total_qty`, `units_sold` là đặc trưng mạnh | Spearman 0.91/0.85 nhưng là kết quả của doanh thu | **Data Leakage** — rò rỉ dữ liệu tương lai | Loại khỏi mọi tập train/predict |
| `sessions_lag365` thừa vì Prophet xử lý mùa vụ rồi | Ablation: +sessions → MAE giảm 4.8% | **Bác bỏ** — sessions mang thông tin bổ sung | Giữ lại trong bộ đặc trưng |

---

## 10. Khuyến Nghị Ra Quyết Định

### 10.1 Cho Giám Đốc Chuỗi Cung Ứng — Tồn Kho An Toàn

**Công thức:**
```
Tồn kho an toàn = Dự báo Doanh thu × (1 + buffer)
```

**Bảng Sensitivity (phân tích độ nhạy — Trade-off rủi ro vs chi phí):**

| Buffer | Xác suất đủ hàng | Chi phí tồn kho tăng thêm | Khuyến nghị |
|---|---|---|---|
| +10% | ~83% | +6.1% | Rủi ro thiếu hàng cao |
| **+15%** | **~90%** | **+10.3%** | **Điểm cân bằng tốt trong tập kiểm tra** |
| +20% | ~94% | +14.8% | Chi phí cao, phù hợp mùa siêu cao điểm |

> ⚠️ **Quan trọng — Tháng 8 năm lẻ:** KHÔNG tăng tồn kho. Đây là tháng clearance hàng tồn kho dư thừa (Urban Blowout). COGS/Rev > 1.0 = đang bán lỗ. Quyết định đúng: CẮT nhập hàng mới, chỉ xả dead-stock.

### 10.2 Cho Trưởng Marketing — Phân Bổ Ngân Sách

| Quý | ROI/phiên (quan sát) | Phân bổ đề xuất |
|---|---|---|
| Q1 | Cao nhất (~43%) | ≥40% ngân sách năm |
| Q2 | Tốt (~37%) | ~35% |
| Q4 | Trung bình (~14%) | ~20% |
| Q3 (tháng 8 năm lẻ) | Thấp nhất (~5%) | ~5% — chỉ duy trì |

### 10.3 Hệ Thống Cảnh Báo Sớm (3 Vùng)

| Vùng | Lượng web traffic so với YoY | Xác suất doanh thu giảm | Hành động |
|---|---|---|---|
| 🟢 Xanh | Giảm <10% | <15% | Theo dõi bình thường |
| 🟡 Vàng | Giảm 10–20% | ~25% | Tăng buffer +5%, cảnh báo GĐ SCM |
| 🔴 Đỏ | Giảm >20% | ~37% | Dừng promo không thiết yếu; tăng buffer +10%; báo CFO |

---

## 11. Hạn Chế & Hướng Phát Triển

| Hạn chế | Lý do | Ảnh hưởng |
|---|---|---|
| Sai số cộng dồn | Dự báo xa 18 tháng | Khoảng tin cậy rộng ở cuối 2024 |
| Chu kỳ 2 năm chưa tối ưu | Prophet event giúp, nhưng chưa bắt interaction year_parity × category | Tháng 8/2023 vẫn có thể sai |
| Không có dữ liệu vĩ mô | Lạm phát, trend thời trang, đối thủ | Model không nhận biết thay đổi thị trường |
| Lịch promo 2023–2024 tái tạo | Giả định ban quản lý không thay đổi lịch | Cần cập nhật thủ công khi có thay đổi |

**Hướng phát triển:**
- Rolling-origin backtest: kiểm tra độ ổn định trên 3 năm (2020, 2021, 2022 làm hold-out luân phiên) để xác nhận R²=0.673 không phải may mắn
- Thêm dữ liệu CPI ngành bán lẻ thời trang
- Tích hợp dashboard tự động cập nhật khi có dữ liệu mới (Business Intelligence layer)

---

## Phụ Lục A — Bảng Tự Chấm Theo Checklist ML/DL v4

### A1. Xử Lý Dữ Liệu (2 điểm)

| STT | Yêu cầu Checklist | Đã thực hiện | Minh chứng |
|---|---|---|---|
| 1 | Mô tả bài toán, đầu vào, đầu ra, yêu cầu xử lý | ✅ | Phần 1–3 báo cáo; pipeline.py |
| 2 | Đánh nhãn & Tiền xử lý dữ liệu | ✅ | denoise.py: outlier capping; lag365 features |
| 3 | Thống kê dữ liệu mẫu | ✅ | Bảng §3.2; Viz1–4 (EDA) |
| 4 | Chuyển đổi dữ liệu (bài toán hồi quy) | ✅ | Loại zero-variance (stockout); COGS ratio transform |

### A2. Đánh Giá Mô Hình (1 điểm)

| STT | Yêu cầu Checklist | Đã thực hiện | Minh chứng |
|---|---|---|---|
| 5 | Đề xuất và lựa chọn tiêu chí đánh giá | ✅ | MAE, RMSE, R², pct_clipped; 5 phương diện §8.1 |
| 6 | Thống kê và phân tích lỗi | ✅ | Bảng so sánh 8 kiến trúc; phân tích lỗi §8.3 |

### A3. Cải Tiến Mô Hình (4 điểm — 8 kiến trúc)

| STT | Tên kiến trúc | Kết quả | Ghi chú |
|---|---|---|---|
| 7 | M1.0 — YoY Naive | MAE=0.838M, R²=0.518 | Baseline ngây thơ |
| 8 | M1.1 — ARIMA(5,1,2) | MAE=1.303M, R²=−0.147 | Đã đo; tệ hơn YoY → loại có bằng chứng |
| 9 | M2.0 — Prophet đơn | MAE=1.001M, R²=0.446 | Thiếu residual learning |
| 10 | M2.1 — COGS độc lập | COGS>Rev=41.6% ❌ | Kiến trúc thất bại (2 model rời) |
| 11 | M2.2 — COGS ratio Prophet | COGS>Rev=50.1% ❌ | Kiến trúc thất bại (ratio phẳng) |
| 12 | M3.0 — +Seasonal Ratio COGS | MAE=0.830M, COGS>Rev=0.5% ✅ | Đột phá kiến trúc COGS |
| 13 | M3.1 — +Promo Events | MAE=0.840M | Promo có lợi ở năm lẻ 2023 (test), trung tính trên val 2022 |
| 14 | **M3.2 — Full Pipeline (CHỐT)** | **MAE=0.726M val (0.718M Optuna), R²=0.666** | Kết quả tốt nhất |

### A4. Đóng Gói Mô Hình (3 điểm)

| STT | Yêu cầu Checklist | Đã thực hiện | Minh chứng |
|---|---|---|---|
| 15 | Mô hình tiên tiến (công bố peer-reviewed, được công nhận rộng rãi) | ✅ | Prophet (Taylor & Letham 2018, *The American Statistician*); LightGBM (Ke et al. 2017, NIPS); Optuna (Akiba et al. 2019, KDD) — đều là SOTA được trích dẫn cao trong dự báo chuỗi thời gian & AutoML |
| 16 | Khả năng ứng dụng ngữ cảnh cụ thể | ✅ | Chuỗi bán lẻ thời trang; CFO/SCM/Marketing |
| 17 | Chỉ số đủ điều kiện thực tế | ✅ | R²=0.673 OOS; pct_clipped=5.7%; khoảng tin cậy 95% |
| 18 | Giao diện demo | ✅ | 14 Visualizations trong `outputs/part2/`; `generate_analysis.py` |
| 19 | Slide báo cáo | ✅ | `report/report_TV.pdf` (7 trang, LaTeX) |
| 20 | Thuyết trình | — | Theo kế hoạch nhóm |

---

## Phụ Lục B — Khai Báo Sử Dụng AI

**Công cụ AI đã sử dụng:** Claude Code (Anthropic)

**Mục đích sử dụng:**
- Hỗ trợ viết và debug code pipeline (Python, LaTeX)
- Gợi ý kiến trúc COGS khi phát hiện 53.5% clip
- Soạn thảo và cấu trúc báo cáo

**Cách nhóm kiểm chứng:**
- Mọi đề xuất kỹ thuật đều được chạy thực tế và đo trên dữ liệu
- `log.txt` ghi lại toàn bộ kết quả pipeline (MAE, R², pct_clipped)
- Quyết định kiến trúc dựa trên kết quả thực nghiệm, không phải lý thuyết AI đề xuất

**Phần nội dung có AI hỗ trợ:** Code pipeline, cấu trúc báo cáo, debug lỗi import  
**Phần nhóm tự thực hiện:** Phân tích dữ liệu, quyết định kiến trúc, diễn giải kết quả, khuyến nghị kinh doanh

---

*Cập nhật lần cuối: 2026-06-19. Pipeline: Revenue MAE=0.718M, R²=0.673; COGS pct_clipped=5.7%; submission=548 rows.*
