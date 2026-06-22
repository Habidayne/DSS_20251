# -*- coding: utf-8 -*-
"""
Gridbreaker DSS — Giao diện Demo Hệ Hỗ Trợ Quyết Định
======================================================
Dự báo Doanh thu & COGS 18 tháng (Prophet + CatBoost, M3.7) và biến dự báo
thành quyết định cho 3 vai trò: CFO, Giám đốc Chuỗi Cung Ứng, Trưởng Marketing.

Chạy:  streamlit run demo/app.py
Dữ liệu đọc trực tiếp từ  submission.csv  (forecast)  +  csv_new/sales.csv  (lịch sử).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình & hằng số (lấy từ GIAI_THICH_DON_GIAN.md — đã kiểm chứng thực nghiệm)
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = ROOT / "submission.csv"
SALES = ROOT / "csv_new" / "sales.csv"
SALES_FALLBACK = ROOT / "csv" / "sales.csv"

FAMI = "#0B5394"      # xanh BK
ORANGE = "#E69138"
GREEN = "#38761D"
RED = "#CC0000"
YELLOW = "#BF9000"

MAPE = 22.4           # % — sai số tuyệt đối trung bình (val 2022, M3.7)
R2_VAL = 0.721        # R² val 2022
CI_MULT = 1.22        # Tồn kho an toàn = Dự báo × 1.22 (CI ~95%)

# Bảng sensitivity tồn kho an toàn (10.1)
BUFFER_TABLE = {
    10: {"fill": 83, "cost": 6.1, "note": "Rủi ro thiếu hàng cao"},
    15: {"fill": 90, "cost": 10.3, "note": "Điểm cân bằng tốt (khuyến nghị)"},
    20: {"fill": 94, "cost": 14.8, "note": "Chi phí cao — chỉ mùa siêu cao điểm"},
}

# Phân bổ ngân sách marketing theo ROI/phiên (10.2)
MKT_ALLOC = {"Q1": 43, "Q2": 37, "Q4": 14, "Q3": 5}
MKT_ROI = {"Q1": "Cao nhất (~43%)", "Q2": "Tốt (~37%)",
           "Q4": "Trung bình (~14%)", "Q3": "Thấp nhất (~5%)"}

st.set_page_config(page_title="Gridbreaker DSS", page_icon="📈", layout="wide")


# ─────────────────────────────────────────────────────────────────────────────
# Tải dữ liệu
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    fc = pd.read_csv(SUBMISSION, parse_dates=["Date"]).sort_values("Date")
    sales_path = SALES if SALES.exists() else SALES_FALLBACK
    hist = pd.read_csv(sales_path, parse_dates=["Date"]).sort_values("Date")
    for d in (fc, hist):
        d["ratio"] = d["COGS"] / d["Revenue"]
        d["margin"] = 1 - d["ratio"]
        d["quarter"] = d["Date"].dt.quarter
        d["year"] = d["Date"].dt.year
        d["month"] = d["Date"].dt.month
    return fc, hist


try:
    forecast, history = load_data()
except FileNotFoundError as e:
    st.error(f"Không tìm thấy dữ liệu: {e}\n\nHãy chạy `python pipeline.py` để tạo submission.csv trước.")
    st.stop()

VND = lambda x: f"{x:,.0f} ₫"
M = lambda x: f"{x/1e6:,.2f}M"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — điều hướng
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Gridbreaker DSS")
    st.caption("Hệ Hỗ Trợ Quyết Định — Dự báo Doanh thu & COGS bán lẻ thời trang")
    view = st.radio(
        "Chọn vai trò / màn hình:",
        ["🏠 Tổng quan", "💰 CFO — Kế hoạch tài chính",
         "📦 GĐ Chuỗi Cung Ứng — Tồn kho", "📣 Trưởng Marketing — Ngân sách",
         "🚦 Cảnh báo sớm (3 vùng)"],
    )
    st.divider()
    st.markdown("**Mô hình:** M3.7 — Prophet (xu hướng + mùa vụ) + CatBoost (phần dư, 23 đặc trưng)")
    st.markdown(f"**R² (val 2022):** {R2_VAL:.3f}  |  **MAPE:** {MAPE:.1f}%")
    st.caption(f"Forecast: {forecast['Date'].min():%d/%m/%Y} → {forecast['Date'].max():%d/%m/%Y} "
               f"({len(forecast)} ngày)")


def kpi_row():
    tot_rev = forecast["Revenue"].sum()
    tot_cogs = forecast["COGS"].sum()
    profit = tot_rev - tot_cogs
    margin = profit / tot_rev * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng Doanh thu dự báo", M(tot_rev), help="18 tháng tới")
    c2.metric("Tổng COGS dự báo", M(tot_cogs))
    c3.metric("Lợi nhuận gộp", M(profit), f"{margin:.1f}% biên")
    c4.metric("Doanh thu TB/ngày", M(forecast["Revenue"].mean()))


# ─────────────────────────────────────────────────────────────────────────────
# 1) TỔNG QUAN
# ─────────────────────────────────────────────────────────────────────────────
if view.startswith("🏠"):
    st.title("Tổng quan — Dự báo 18 tháng")
    st.markdown("Mỗi ngày trong 18 tháng tới sẽ có **bao nhiêu doanh thu và chi phí hàng bán**, "
                "với mức độ không chắc chắn nào? — và điều đó dẫn tới quyết định gì.")
    kpi_row()
    st.divider()

    show_hist = st.toggle("Hiện lịch sử (2012–2022)", value=True)
    show_ci = st.toggle("Hiện dải tin cậy ~95% (Dự báo × 1.22)", value=True)

    fig = go.Figure()
    if show_hist:
        h = history[history["Date"] >= "2020-01-01"]
        fig.add_trace(go.Scatter(x=h["Date"], y=h["Revenue"], name="Doanh thu (lịch sử)",
                                 line=dict(color="lightgray", width=1)))
    if show_ci:
        fig.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Revenue"] * CI_MULT,
                                 name="Cận trên 95%", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Revenue"] * (2 - CI_MULT),
                                 name="Dải tin cậy ±22%", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(11,83,148,0.15)"))
    fig.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Revenue"], name="Doanh thu (dự báo)",
                             line=dict(color=FAMI, width=2)))
    fig.add_trace(go.Scatter(x=forecast["Date"], y=forecast["COGS"], name="COGS (dự báo)",
                             line=dict(color=ORANGE, width=1.5, dash="dot")))
    fig.update_layout(height=460, hovermode="x unified",
                      legend=dict(orientation="h", y=1.08),
                      yaxis_title="VND/ngày", margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")

    st.info("**Đọc biểu đồ:** Doanh thu dự báo dao động theo mùa vụ (đỉnh Q2, đáy Q4). "
            "Dải xanh nhạt = khoảng tin cậy ~95% dùng để tính tồn kho an toàn. "
            "Những ngày COGS (cam) chạm/vượt Doanh thu = **tháng 8 năm lẻ (clearance Urban Blowout)** — bán lỗ có chủ đích.")


# ─────────────────────────────────────────────────────────────────────────────
# 2) CFO — Kế hoạch tài chính theo quý
# ─────────────────────────────────────────────────────────────────────────────
elif view.startswith("💰"):
    st.title("💰 CFO — Kế hoạch tài chính theo quý")
    kpi_row()
    st.divider()

    fc = forecast.copy()
    fc["period"] = fc["Date"].dt.to_period("Q").astype(str)
    q = fc.groupby("period").agg(Revenue=("Revenue", "sum"), COGS=("COGS", "sum")).reset_index()
    q["Lợi nhuận gộp"] = q["Revenue"] - q["COGS"]
    q["Biên (%)"] = (q["Lợi nhuận gộp"] / q["Revenue"] * 100).round(1)

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_bar(x=q["period"], y=q["Revenue"] / 1e6, name="Doanh thu", marker_color=FAMI)
        fig.add_bar(x=q["period"], y=q["COGS"] / 1e6, name="COGS", marker_color=ORANGE)
        fig.add_trace(go.Scatter(x=q["period"], y=q["Lợi nhuận gộp"] / 1e6, name="Lợi nhuận gộp",
                                 line=dict(color=GREEN, width=3), mode="lines+markers"))
        fig.update_layout(height=420, barmode="group", yaxis_title="Triệu VND",
                          legend=dict(orientation="h", y=1.1), margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")
    with right:
        disp = q.copy()
        for c in ["Revenue", "COGS", "Lợi nhuận gộp"]:
            disp[c] = (disp[c] / 1e6).round(1)
        disp.columns = ["Quý", "Doanh thu (M)", "COGS (M)", "LN gộp (M)", "Biên (%)"]
        st.dataframe(disp, width="stretch", hide_index=True)

    # Cảnh báo COGS > Revenue (tháng 8 năm lẻ)
    viol = forecast[forecast["COGS"] > forecast["Revenue"]]
    if len(viol):
        st.warning(f"⚠️ **{len(viol)} ngày COGS > Doanh thu** ({len(viol)/len(forecast)*100:.1f}% horizon) — "
                   f"rơi vào **tháng 8/2023 (năm lẻ)**, đợt clearance Urban Blowout (bán lỗ có chủ đích để xả tồn kho). "
                   f"Khuyến nghị CFO: siết ngân sách marketing tháng 8 năm lẻ, không nhập hàng mới.")
    st.caption("Biên lợi nhuận mỏng (~13–15%) → sai số dự báo nhỏ cũng tác động lớn. Đó là lý do mô hình "
               "ràng buộc COGS = Doanh thu × tỷ lệ thay vì dự báo 2 chỉ số độc lập.")


# ─────────────────────────────────────────────────────────────────────────────
# 3) GĐ CHUỖI CUNG ỨNG — Tồn kho an toàn
# ─────────────────────────────────────────────────────────────────────────────
elif view.startswith("📦"):
    st.title("📦 Giám đốc Chuỗi Cung Ứng — Tồn kho an toàn")
    st.markdown("**Công thức:** `Tồn kho an toàn = Dự báo Doanh thu × (1 + buffer)`")

    buffer = st.select_slider("Chọn mức buffer an toàn:", options=[10, 15, 20], value=15)
    info = BUFFER_TABLE[buffer]
    c1, c2, c3 = st.columns(3)
    c1.metric("Xác suất đủ hàng", f"~{info['fill']}%")
    c2.metric("Chi phí tồn kho tăng thêm", f"+{info['cost']}%")
    c3.metric("Khuyến nghị", "✅ Tốt" if buffer == 15 else ("⚠️ Cao rủi ro" if buffer == 10 else "💸 Đắt"))
    st.caption(info["note"])
    st.divider()

    # Kế hoạch đặt hàng theo tháng
    fc = forecast.copy()
    fc["period"] = fc["Date"].dt.to_period("M").astype(str)
    mth = fc.groupby("period").agg(Revenue=("Revenue", "sum")).reset_index()
    mth["Tồn kho an toàn"] = mth["Revenue"] * (1 + buffer / 100)

    fig = go.Figure()
    fig.add_bar(x=mth["period"], y=mth["Revenue"] / 1e6, name="Doanh thu dự báo", marker_color=FAMI)
    fig.add_trace(go.Scatter(x=mth["period"], y=mth["Tồn kho an toàn"] / 1e6,
                             name=f"Mức chuẩn bị hàng (+{buffer}%)",
                             line=dict(color=ORANGE, width=3), mode="lines+markers"))
    fig.update_layout(height=400, yaxis_title="Triệu VND", xaxis_tickangle=-45,
                      legend=dict(orientation="h", y=1.1), margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

    # Bảng sensitivity
    st.subheader("Bảng phân tích độ nhạy (rủi ro ↔ chi phí)")
    sens = pd.DataFrame([
        {"Buffer": f"+{b}%", "Xác suất đủ hàng": f"~{v['fill']}%",
         "Chi phí tăng thêm": f"+{v['cost']}%", "Đánh giá": v["note"]}
        for b, v in BUFFER_TABLE.items()])
    st.dataframe(sens, width="stretch", hide_index=True)

    st.error("🚫 **Tháng 8 năm lẻ — KHÔNG tăng tồn kho.** Đây là tháng clearance hàng dư thừa "
             "(Urban Blowout), COGS/Doanh thu > 1.0 = đang bán lỗ. Quyết định đúng: **cắt nhập hàng mới, "
             "chỉ xả dead-stock.** Tăng buffer chỉ áp dụng cho **nhóm bán chạy mùa cao điểm Q1–Q2.**")


# ─────────────────────────────────────────────────────────────────────────────
# 4) TRƯỞNG MARKETING — Phân bổ ngân sách
# ─────────────────────────────────────────────────────────────────────────────
elif view.startswith("📣"):
    st.title("📣 Trưởng Marketing — Phân bổ ngân sách theo quý")
    st.markdown("Phân bổ dựa trên **ROI marketing/phiên** (không phải theo doanh thu tuyệt đối). "
                "Q1 có ROI/phiên cao nhất dù doanh thu tuyệt đối thấp hơn Q2.")

    budget = st.number_input("Tổng ngân sách marketing năm (triệu VND):",
                             min_value=100, max_value=100000, value=2000, step=100)
    st.divider()

    alloc = pd.DataFrame([
        {"Quý": q, "ROI/phiên (quan sát)": MKT_ROI[q], "Tỷ lệ đề xuất (%)": p,
         "Ngân sách (triệu VND)": round(budget * p / 100, 1)}
        for q, p in sorted(MKT_ALLOC.items())])

    left, right = st.columns([2, 3])
    with left:
        fig = go.Figure(go.Pie(labels=alloc["Quý"], values=alloc["Tỷ lệ đề xuất (%)"],
                               marker_colors=[FAMI, ORANGE, RED, GREEN], hole=0.45,
                               textinfo="label+percent"))
        fig.update_layout(height=360, margin=dict(t=10, b=10),
                          annotations=[dict(text="Ngân sách", showarrow=False)])
        st.plotly_chart(fig, width="stretch")
    with right:
        st.dataframe(alloc, width="stretch", hide_index=True)
        st.success(f"**Khuyến nghị:** dồn **≥40%** ngân sách (~{budget*0.43:,.0f}M) vào **Q1** — "
                   f"ROI/phiên cao nhất. Q3 giữ ~5% (chỉ duy trì), đặc biệt tháng 8 năm lẻ.")

    st.caption("⚠️ Nghịch lý Promotion (Simpson): promo trùng mùa cao điểm trông như tăng doanh thu, "
               "nhưng Event Study cho thấy uplift ròng đã hiệu chỉnh mùa vụ là **âm (~−17%)**. "
               "→ Ưu tiên promo **theo %** thay vì giảm giá cố định.")


# ─────────────────────────────────────────────────────────────────────────────
# 5) CẢNH BÁO SỚM — 3 vùng
# ─────────────────────────────────────────────────────────────────────────────
elif view.startswith("🚦"):
    st.title("🚦 Hệ thống cảnh báo sớm — 3 vùng")
    st.markdown("Lưu lượng web traffic **không dẫn trước doanh thu** (CCF ≈ 0.32 ở mọi độ trễ), "
                "nhưng dùng tốt làm **chỉ báo YoY** để cảnh báo sớm rủi ro sụt doanh thu.")

    yoy = st.slider("Web traffic so với cùng kỳ năm ngoái (% thay đổi):",
                    min_value=-50, max_value=20, value=-5, step=1)

    if yoy >= -10:
        zone, color, prob, action = ("🟢 XANH", GREEN, "<15%", "Theo dõi bình thường.")
    elif yoy >= -20:
        zone, color, prob, action = ("🟡 VÀNG", YELLOW, "~25%",
                                     "Tăng buffer tồn kho +5%; cảnh báo Giám đốc Chuỗi Cung Ứng.")
    else:
        zone, color, prob, action = ("🔴 ĐỎ", RED, "~37%",
                                     "Dừng promo không thiết yếu; tăng buffer +10%; báo cáo CFO.")

    st.markdown(f"<h2 style='color:{color}'>{zone}</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Xác suất doanh thu giảm", prob)
    c2.metric("Traffic YoY", f"{yoy:+d}%")
    st.markdown(f"### Hành động đề xuất\n> {action}")
    st.divider()

    zones = pd.DataFrame([
        {"Vùng": "🟢 Xanh", "Traffic YoY": "Giảm <10%", "Xác suất giảm DT": "<15%", "Hành động": "Theo dõi bình thường"},
        {"Vùng": "🟡 Vàng", "Traffic YoY": "Giảm 10–20%", "Xác suất giảm DT": "~25%", "Hành động": "Tăng buffer +5%, cảnh báo SCM"},
        {"Vùng": "🔴 Đỏ", "Traffic YoY": "Giảm >20%", "Xác suất giảm DT": "~37%", "Hành động": "Dừng promo, buffer +10%, báo CFO"},
    ])
    st.dataframe(zones, width="stretch", hide_index=True)


st.divider()
st.caption("Gridbreaker DSS · Datathon 2026 Vòng 1 · Mô hình M3.7 (Prophet + CatBoost) · "
           "Dữ liệu đọc trực tiếp từ submission.csv & csv_new/sales.csv")
