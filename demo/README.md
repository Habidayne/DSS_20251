# Gridbreaker DSS — Giao diện Demo

Giao diện web demo cho **Hệ Hỗ Trợ Quyết Định** dự báo Doanh thu & COGS 18 tháng
(mô hình **M3.7 — Prophet + CatBoost**). Đáp ứng tiêu chí *"Đóng gói giao diện
demo chương trình"* trong checklist.

Demo **đọc trực tiếp** kết quả dự báo từ `../submission.csv` và lịch sử từ
`../csv_new/sales.csv` — không hard-code số liệu.

## Cách chạy

**Windows (1 click):** double-click `run_demo.bat`

**Thủ công:**
```bash
pip install -r requirements.txt
streamlit run demo/app.py        # chạy từ thư mục gốc dự án
```
Giao diện mở tại http://localhost:8501

## 5 màn hình (theo 3 vai trò ra quyết định)

| Màn hình | Vai trò | Quyết định hỗ trợ |
|---|---|---|
| 🏠 Tổng quan | Tất cả | Dự báo Doanh thu/COGS + dải tin cậy ~95% |
| 💰 CFO | Giám đốc Tài chính | Kế hoạch tài chính theo quý, cảnh báo tháng 8 năm lẻ |
| 📦 GĐ Chuỗi Cung Ứng | Vận hành kho | Tồn kho an toàn = Dự báo × (1+buffer), bảng sensitivity |
| 📣 Trưởng Marketing | Marketing | Phân bổ ngân sách theo ROI/phiên (Q1 ≥40%) |
| 🚦 Cảnh báo sớm | CFO + SCM | 3 vùng (Xanh/Vàng/Đỏ) theo traffic YoY |

## Phụ thuộc
`streamlit`, `pandas`, `plotly` — xem `requirements.txt`.
