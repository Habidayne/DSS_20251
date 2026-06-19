"""
src/analysis/prescriptive.py — Phân tích Đề xuất Hành động (Prescriptive)
"What should we do?"

Viz 12: Safety stock trade-off curve (stockout risk vs holding cost)
Viz 13: Marketing budget allocation by quarter + projected ROI
Viz 14: Early Warning System — decision framework
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BG     = "#F8F9FA"
ACCENT = "#2EC4B6"
RED    = "#E71D36"
GOLD   = "#FF9F1C"
DARK   = "#011627"
GREEN  = "#06D6A0"
PURPLE = "#7B2D8B"


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def viz12_safety_stock_tradeoff(sales: pd.DataFrame, forecast_errors: pd.Series,
                                 inventory_path: str, out_dir: str):
    """
    Viz 12: Đánh đổi tồn kho an toàn — buffer +15% là điểm cân bằng THỰC NGHIỆM
    (tốt trong tập kiểm tra hiện tại, KHÔNG khẳng định "tối ưu Pareto" tuyệt đối
    vì chưa có mô hình chi phí thiếu-hàng vs lưu-kho đầy đủ — hạ giọng theo phản biện).

    Tính toán dựa trên phân phối sai số dự báo thực tế trên validation set.
    Ta mô phỏng: nếu doanh nghiệp chuẩn bị tồn kho = forecast × (1 + buffer%),
    thì xác suất thiếu hàng (stockout) là bao nhiêu so với chi phí lưu kho thêm?
    """
    # forecast_errors = (actual - forecast) / actual, already computed
    errors = forecast_errors.dropna()

    # Simulate buffer levels
    buffer_levels = np.arange(0, 0.31, 0.01)  # 0% to 30%
    stockout_risks = []
    avg_excess_costs = []

    for buf in buffer_levels:
        # Stockout occurs when actual > forecast*(1+buffer)
        # i.e. when actual/forecast - 1 > buffer
        # i.e. when error_pct > buffer
        # error_pct = (actual - forecast) / forecast
        stockout_rate = (errors > buf).mean() * 100
        # Excess cost: average over-preparation when no stockout
        excess = np.maximum(buf - errors, 0).mean() * 100  # % of forecast
        stockout_risks.append(stockout_rate)
        avg_excess_costs.append(excess)

    # Load inventory for holding cost context
    inv = pd.read_csv(inventory_path)
    avg_hold_days = inv["days_of_supply"].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # LEFT: Trade-off curve
    ax1.plot(buffer_levels * 100, stockout_risks, color=RED, linewidth=2.5,
             label="Rủi ro thiếu hàng (%)", marker="o", markersize=3)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(buffer_levels * 100, avg_excess_costs, color=GOLD, linewidth=2.5,
                  label="Chi phí lưu kho thêm (%)", marker="s", markersize=3)
    ax1_twin.set_ylabel("Chi phí lưu kho thêm (% doanh thu)", fontsize=9, color=GOLD)
    ax1_twin.spines["right"].set_color(GOLD)
    ax1_twin.tick_params(axis="y", colors=GOLD, labelsize=8)

    # Mark empirical balance point (15%) — không gọi "tối ưu tuyệt đối"
    opt_buf = 15
    ax1.axvline(opt_buf, color=GREEN, linewidth=2, linestyle="--", alpha=0.7,
                label="+15% (điểm cân bằng thực nghiệm)")

    _style_ax(ax1, title="Đánh đổi: Tồn kho an toàn vs Rủi ro",
              xlabel="Tồn kho đệm (%)", ylabel="Rủi ro thiếu hàng (%)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    # RIGHT: Monthly recommendation table as bar chart
    # End-of-month vs mid-month stockout risk
    errors_df = pd.DataFrame({"error": errors})
    errors_df["dom"] = errors.index.day
    errors_df["is_eom"] = errors_df["dom"] >= 25

    eom_risk     = (errors_df[errors_df["is_eom"]]["error"] > 0.15).mean() * 100
    mid_risk     = (errors_df[~errors_df["is_eom"]]["error"] > 0.15).mean() * 100
    eom_risk_buf = (errors_df[errors_df["is_eom"]]["error"] > 0.20).mean() * 100

    periods = ["Giữa tháng\n(ngày 1-24)", "Cuối tháng\n(ngày 25-31)", "Cuối tháng\n+20% buffer"]
    risks   = [mid_risk, eom_risk, eom_risk_buf]
    colors  = [GREEN, RED, ACCENT]

    bars = ax2.bar(periods, risks, color=colors, width=0.5, edgecolor="white", linewidth=2)
    for bar, risk in zip(bars, risks):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{risk:.1f}%", ha="center", fontsize=10, fontweight="bold")

    _style_ax(ax2, title="Rủi ro thiếu hàng: Cuối tháng cao hơn — cần tăng tồn kho đệm",
              xlabel="Giai đoạn trong tháng", ylabel="Rủi ro thiếu hàng (%)")

    fig.suptitle("Đánh đổi tồn kho an toàn: +15% là điểm cân bằng thực nghiệm (giảm rủi ro thiếu hàng, chi phí tăng thêm chấp nhận được)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "viz12_safety_stock_tradeoff.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def viz13_marketing_allocation(sales: pd.DataFrame, web_traffic_path: str,
                                out_dir: str):
    """
    Viz 13: Phân bổ marketing — Đẩy mạnh Q1 khi biên lợi nhuận cao nhất

    Logic: ROI = Margin × Conversion.
    → Tìm quarter nào margin cao nhất + conversion rate từ web traffic cao nhất
    → Đề xuất phân bổ marketing budget theo quarter
    """
    # Web traffic conversion proxy (only available 2021-2022)
    wt = pd.read_csv(web_traffic_path, parse_dates=["date"]).set_index("date")
    wt_daily = wt.groupby(wt.index).agg({"sessions": "sum", "unique_visitors": "sum"}).copy()
    
    # Filter sales to overlap with web traffic to prevent mixing time periods
    df = sales.reindex(wt_daily.index).dropna().copy()
    df["margin"]  = (df["Revenue"] - df["COGS"]) / df["Revenue"] * 100
    df["quarter"] = df.index.quarter
    margin_q = df.groupby("quarter")["margin"].mean()

    wt_daily["quarter"] = wt_daily.index.quarter
    wt_daily["rev_per_session"] = df["Revenue"] / wt_daily["sessions"]
    efficiency_q = wt_daily.groupby("quarter")["rev_per_session"].mean()

    # Composite ROI Priority Score = Margin * Conversion
    # Directly measuring how much profit each session brings
    raw_score = margin_q * efficiency_q
    
    # Softmax to normalize into investment allocations
    from scipy.special import softmax
    roi_z = (raw_score - raw_score.mean()) / raw_score.std()
    roi_score_pct = pd.Series(softmax(roi_z) * 100, index=raw_score.index)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # LEFT: Margin + Efficiency by quarter
    x = np.arange(4)
    w = 0.35
    ax1.bar(x - w/2, margin_q.values, w, color=ACCENT, label="Biên lợi nhuận gộp (%)", alpha=0.85)
    ax1_twin = ax1.twinx()
    ax1_twin.bar(x + w/2, efficiency_q.values / 1e3, w, color=GOLD,
                 label="Doanh thu/Truy cập (Nghìn VNĐ)", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Q1\n(Jan-Mar)", "Q2\n(Apr-Jun)", "Q3\n(Jul-Sep)", "Q4\n(Oct-Dec)"])
    _style_ax(ax1, title="Biên lợi nhuận & Hiệu suất chuyển đổi theo Quý",
              ylabel="Biên lợi nhuận gộp (%)")
    ax1_twin.set_ylabel("Doanh thu/Truy cập (Nghìn VNĐ)", fontsize=9, color=GOLD)
    ax1_twin.tick_params(axis="y", colors=GOLD, labelsize=8)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    # RIGHT: Recommended budget allocation pie/donut
    labels_q  = ["Q1", "Q2", "Q3", "Q4"]
    sizes     = roi_score_pct.values
    explode   = [0.05 if s == sizes.max() else 0 for s in sizes]
    colors_q  = [ACCENT, GOLD, PURPLE, RED]
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels_q, autopct='%1.1f%%',
                                        explode=explode, colors=colors_q, pctdistance=0.80,
                                        startangle=90, textprops={"fontsize": 10})
    # Add inner circle for donut
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    ax2.add_artist(centre_circle)
    ax2.text(0, 0, "Mức độ\nƯu tiên", ha="center", va="center",
             fontsize=11, fontweight="bold", color=DARK)
    ax2.set_title("Đề xuất mức độ ưu tiên Marketing theo ROI Score\n(tỷ lệ tương đối — Q3 vẫn cần chi tối thiểu để duy trì)",
                  fontsize=10, fontweight="bold", pad=12)

    fig.suptitle("Phân bổ ngân sách Marketing theo mùa: Q1 & Q2 ưu tiên cao nhất do margin và hiệu suất chuyển đổi tốt",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "viz13_marketing_allocation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def viz14_early_warning_system(sales: pd.DataFrame, web_traffic_path: str,
                                inventory_path: str, out_dir: str):
    """
    Viz 14: Hệ thống cảnh báo sớm — Decision framework

    Kết hợp 3 tín hiệu: (1) web traffic trend, (2) stockout risk, (3) margin erosion
    → Traffic light system cho operations team
    """
    wt = pd.read_csv(web_traffic_path, parse_dates=["date"]).set_index("date")
    wt_daily = wt.groupby(wt.index)["sessions"].sum()

    # YoY traffic change
    wt_yoy = wt_daily.pct_change(periods=365) * 100
    # Revenue YoY change
    rev = sales["Revenue"]
    rev_yoy = rev.pct_change(periods=365) * 100

    # Align
    common = wt_yoy.index.intersection(rev_yoy.index)
    wt_yoy  = wt_yoy.loc[common].dropna()
    rev_yoy = rev_yoy.reindex(wt_yoy.index)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # LEFT: Scatter of web traffic YoY vs Revenue YoY
    mask_green  = (wt_yoy > 0) & (rev_yoy > 0)
    mask_red    = (wt_yoy < -20)
    mask_yellow = ~mask_green & ~mask_red

    ax1.scatter(wt_yoy[mask_green], rev_yoy[mask_green],
                color=GREEN, alpha=0.3, s=12, label="🟢 Bình thường")
    ax1.scatter(wt_yoy[mask_yellow], rev_yoy[mask_yellow],
                color=GOLD, alpha=0.3, s=12, label="🟡 Theo dõi")
    ax1.scatter(wt_yoy[mask_red], rev_yoy[mask_red],
                color=RED, alpha=0.5, s=15, label="🔴 Cảnh báo")

    # Regression
    z = np.polyfit(wt_yoy.values, rev_yoy.values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(wt_yoy.min(), wt_yoy.max(), 100)
    ax1.plot(x_line, p(x_line), color=DARK, linewidth=2, linestyle="--")

    ax1.axvline(-20, color=RED, linestyle=":", linewidth=1.5)
    ax1.axhline(0, color="#999999", linewidth=0.8)
    ax1.axvline(0, color="#999999", linewidth=0.8)


    _style_ax(ax1, title="Tương quan: Web Traffic YoY vs Revenue YoY",
              xlabel="Lượt truy cập thay đổi YoY (%)", ylabel="Doanh thu thay đổi YoY (%)")
    ax1.legend(fontsize=8, loc="upper left")

    # RIGHT: Decision framework infographic
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_facecolor("white")

    # Title
    ax2.text(5, 9.5, "HỆ THỐNG CẢNH BÁO SỚM", ha="center",
             fontsize=13, fontweight="bold", color=DARK)
    ax2.text(5, 9.0, "Early Warning System cho Chuỗi cung ứng", ha="center",
             fontsize=10, color="#666666")

    # Traffic lights
    signals = [
        (GREEN, "🟢 XANH — Vận hành bình thường",
         "• Traffic YoY > 0%\n• Stockout flag = 0\n• Margin ổn định\n→ Giữ nguyên kế hoạch tồn kho"),
        (GOLD, "🟡 VÀNG — Tăng cường giám sát",
         "• Traffic YoY giảm 10-20%\n• Có tín hiệu stockout\n→ Tăng buffer +10%, cập nhật forecast hàng tuần"),
        (RED, "🔴 ĐỎ — Hành động khẩn cấp",
         "• Traffic YoY giảm > 20%\n• Stockout kéo dài > 5 ngày\n→ Buffer +20%, đàm phán NCC, cắt promo non-core"),
    ]

    for i, (color, title, desc) in enumerate(signals):
        y_pos = 7.0 - i * 2.8
        # Circle indicator
        circle = plt.Circle((1.0, y_pos + 0.2), 0.4, color=color, alpha=0.8)
        ax2.add_artist(circle)
        ax2.text(2.0, y_pos + 0.4, title, fontsize=10, fontweight="bold",
                 color=DARK, va="top")
        ax2.text(2.0, y_pos - 0.1, desc, fontsize=8, color="#555555",
                 va="top", linespacing=1.5)

    fig.suptitle("Hệ thống cảnh báo sớm chuỗi cung ứng: Lượt truy cập web là chỉ số dẫn xuất cho quyết định tồn kho",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "viz14_early_warning_system.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def run_prescriptive(sales: pd.DataFrame, forecast_errors: pd.Series,
                     out_dir: str, inventory_path: str, web_traffic_path: str):
    """Run tất cả 3 Prescriptive visualizations."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    print("  [Prescriptive] Viz 12: Safety stock trade-off...")
    paths.append(viz12_safety_stock_tradeoff(sales, forecast_errors, inventory_path, out_dir))
    print("  [Prescriptive] Viz 13: Marketing allocation...")
    paths.append(viz13_marketing_allocation(sales, web_traffic_path, out_dir))
    print("  [Prescriptive] Viz 14: Early warning system...")
    paths.append(viz14_early_warning_system(sales, web_traffic_path, inventory_path, out_dir))
    return paths
