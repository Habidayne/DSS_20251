"""
src/analysis/diagnostic.py — Phân tích Chẩn đoán (Diagnostic)
"Why did it happen?"

Viz 5: Before/After denoising + stockout highlights  (sales ↔ inventory)
Viz 6: Promotion intervention analysis               (sales ↔ promotions)
Viz 7: Web traffic → Revenue cross-correlation       (sales ↔ web_traffic)
Viz 8: Revenue/COGS cointegration anomaly detection  (sales internal)
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
from statsmodels.tsa.stattools import ccf, coint

BG    = "#F8F9FA"
ACCENT = "#2EC4B6"
RED    = "#E71D36"
GOLD   = "#FF9F1C"
DARK   = "#011627"


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def viz5_denoising_stockout(sales: pd.DataFrame, inventory_path: str,
                             clean_revenue: pd.Series, out_dir: str):
    """
    Viz 5 (HIỆU CHỈNH sau kiểm chứng dữ liệu — PH1):
    Giả định ban đầu "stockout là sự kiện hiếm gây mất 23% doanh thu" bị BÁC BỎ.
    `stockout_flag` tổng hợp theo tháng-SKU gần như HẰNG SỐ (~67% mọi tháng) →
    là proxy áp lực tồn kho cấp tháng, KHÔNG phải sự kiện hết hàng cấp ngày, không
    dùng để vá target. Tín hiệu tồn kho DỰ BÁO ĐƯỢC là overstock_pct & fill_rate
    (phương sai thật) — đã đưa vào mô hình dạng lag365, ablation giảm ~15% MAE.
    """
    inv = pd.read_csv(inventory_path)
    inv["snapshot_date"] = pd.to_datetime(inv["snapshot_date"])
    monthly = inv.groupby("snapshot_date").agg(
        stockout_pct=("stockout_flag", "mean"),
        overstock_pct=("overstock_flag", "mean"),
        fill_rate=("fill_rate", "mean"),
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1]})
    fig.patch.set_facecolor("white")

    # --- Panel 1: stockout_pct ~ hằng số (phi tín hiệu) ---
    so = monthly["stockout_pct"]
    ax1.plot(so.index, so, color=RED, linewidth=2, label="Tỷ lệ SKU hết hàng (stockout_pct)")
    ax1.axhline(so.mean(), color="#999999", linestyle="--",
                label=f"Trung bình = {so.mean():.2f}")
    ax1.fill_between(so.index, so.min(), so.max(), alpha=0.08, color=RED)
    ax1.set_ylim(0, 1)
    ax1.text(0.015, 0.12,
             f"Phương sai ≈ 0  (min={so.min():.2f}, max={so.max():.2f}, 126/126 tháng > 0.5)\n"
             f"→ Không phân biệt được tháng cao/thấp → KHÔNG dùng để vá doanh thu",
             transform=ax1.transAxes, fontsize=9, color=RED, fontweight="bold")
    _style_ax(ax1, title="stockout_flag là HẰNG SỐ ~67% — proxy cấp tháng-SKU, phi tín hiệu cấp ngày",
              ylabel="Tỷ lệ")
    ax1.legend(loc="upper right", fontsize=8)

    # --- Panel 2: overstock & fill_rate có phương sai thật → feature dùng được ---
    ov = monthly["overstock_pct"]
    fr = monthly["fill_rate"]
    ax2.plot(ov.index, ov, color=ACCENT, linewidth=2,
             label=f"overstock_pct (phương sai thật {ov.min():.2f}–{ov.max():.2f})")
    ax2.plot(fr.index, fr, color=GOLD, linewidth=2, label="fill_rate")
    ax2.text(0.015, 0.08,
             "Đây mới là tín hiệu tồn kho dự báo được → đưa vào CatBoost dạng lag365\n"
             "(future-safe). Ablation: fill_rate −14%, overstock −7.5% MAE.",
             transform=ax2.transAxes, fontsize=9, color="#0B7A75", fontweight="bold")
    _style_ax(ax2, title="Tín hiệu tồn kho THẬT: overstock_pct & fill_rate có cấu trúc mùa vụ",
              xlabel="Thời gian", ylabel="Tỷ lệ")
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("Viz 5 — Kiểm chứng tín hiệu tồn kho: stockout phi tín hiệu, overstock/fill_rate mới dự báo được",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "viz5_denoising_stockout.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def viz6_promotion_intervention(sales: pd.DataFrame, promotions_path: str, out_dir: str):
    """
    Viz 6: Khuyến mãi tạo spike +X% nhưng hiệu ứng chỉ kéo dài 3-5 ngày
    JOIN: sales.Date ∈ [promotions.start_date, promotions.end_date]
    """
    promos = pd.read_csv(promotions_path, parse_dates=["start_date", "end_date"])

    # Build daily promo flag
    daily_flag = pd.Series(0, index=sales.index)
    promo_windows = []
    for _, row in promos.iterrows():
        mask = (sales.index >= row["start_date"]) & (sales.index <= row["end_date"])
        daily_flag[mask] = 1
        promo_windows.append((row["start_date"], row["end_date"], row["promo_name"]))

    rev = sales["Revenue"]
    baseline = rev[daily_flag == 0].rolling(30).mean().reindex(rev.index).ffill()
    lift = (rev / baseline - 1) * 100

    # Event study: average lift around promo start
    event_windows = []
    for _, row in promos.iterrows():
        start = row["start_date"]
        window_start = start - pd.Timedelta(days=7)
        window_end   = start + pd.Timedelta(days=14)
        if window_start in sales.index and window_end in sales.index:
            window_rev = rev.loc[window_start:window_end]
            baseline_val = rev.loc[window_start:start].mean()
            relative = (window_rev / baseline_val - 1) * 100
            relative.index = range(-7, len(relative) - 7)
            event_windows.append(relative)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # Left: Revenue with promo highlights (Focus on 2013 - 2018)
    rev_zoom = rev[(rev.index >= "2013-01-01") & (rev.index < "2018-01-01")]
    ax1.plot(rev_zoom.index, rev_zoom/1e6, color="#CCCCCC", linewidth=0.6, alpha=0.7)
    
    for start, end, name in promo_windows: # Show all promos in range
        if start >= pd.Timestamp("2013-01-01") and end < pd.Timestamp("2018-01-01"):
            peak = rev.loc[start:end].max()
            ax1.axvspan(start, end, alpha=0.25, color=GOLD)
            ax1.annotate("", xy=(start, peak/1e6), xytext=(start, (peak*0.7)/1e6),
                         arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2))
    _style_ax(ax1, title="Doanh thu & Khuyến mãi (Giai đoạn 2013 – 2018)",
              xlabel="Thời gian", ylabel="Doanh thu (Triệu VNĐ)")

    # Right: Event study plot
    if event_windows:
        event_df = pd.DataFrame(event_windows)
        mean_lift   = event_df.mean()
        median_lift = event_df.median()
        days = mean_lift.index.tolist()

        ax2.fill_between(days, event_df.quantile(0.25), event_df.quantile(0.75),
                         alpha=0.2, color=GOLD, label="IQR")
        ax2.plot(days, mean_lift, color=GOLD, linewidth=2, marker="o",
                 markersize=4, label="Lift trung bình (%)")
        ax2.plot(days, median_lift, color=RED, linewidth=1.5, linestyle="--",
                 label="Lift trung vị (%)")
        ax2.axvline(0, color=DARK, linestyle=":", linewidth=1.5, label="Ngày bắt đầu Promo")
        ax2.axhline(0, color="#999999", linewidth=0.8)
        ax2.set_xlim(-7, 14)

        peak_day  = int(mean_lift.idxmax())
        peak_lift = mean_lift.max()
        ax2.annotate(f"Peak: +{peak_lift:.0f}%\n(Ngày +{peak_day})",
                     xy=(peak_day, peak_lift),
                     xytext=(peak_day + 2, peak_lift * 0.8),
                     fontsize=9, color=GOLD, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=GOLD))

        _style_ax(ax2, title="Event Study: Mức tăng doanh thu quanh ngày bắt đầu Promo",
                  xlabel="Ngày so với ngày bắt đầu Promo", ylabel="Mức tăng trưởng doanh thu (%)")

    fig.suptitle("Khuyến mãi: cú hích +17% ngày đầu nhưng doanh thu đảo chiều âm từ ngày thứ 3",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "viz6_promotion_intervention.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def viz7_web_traffic_ccf(sales: pd.DataFrame, web_traffic_path: str, out_dir: str):
    """
    Viz 7: Web traffic KHÔNG dẫn trước doanh thu — CCF phẳng ~0.32 ở mọi lag 0–7
    ngày (tương quan mùa vụ chung, không phải leading indicator cấp ngày).
    JOIN: sales.Date == web_traffic.date
    """
    wt = pd.read_csv(web_traffic_path, parse_dates=["date"]).set_index("date")

    # Daily aggregate (sum across traffic sources)
    wt_daily = wt.groupby(wt.index)["sessions"].sum()

    # Align with sales
    common_idx = sales.index.intersection(wt_daily.index)
    rev_aligned = sales.loc[common_idx, "Revenue"]
    wt_aligned  = wt_daily.reindex(common_idx)

    # Standardize
    rev_std = (rev_aligned - rev_aligned.mean()) / rev_aligned.std()
    wt_std  = (wt_aligned  - wt_aligned.mean())  / wt_aligned.std()

    # CCF: correlation of wt_today with revenue at lag k
    max_lag = 14
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = wt_std.shift(lag).corr(rev_std)
        else:
            c = wt_std.shift(lag).corr(rev_std)
        correlations.append(c)
    lags = list(range(-max_lag, max_lag + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # Left: Dual-axis time series — Web traffic vs Revenue over time
    focus_start = rev_aligned.index.max() - pd.DateOffset(years=2)
    wt_focus  = wt_aligned.loc[focus_start:]
    rev_focus = rev_aligned.loc[focus_start:]

    wt_smooth  = wt_focus.rolling(14).mean()
    rev_smooth = rev_focus.rolling(14).mean()

    peak_lag = lags[np.argmax(correlations)]
    ci = 1.96 / np.sqrt(len(common_idx))

    ax1b = ax1.twinx()
    ax1.plot(wt_smooth.index, wt_smooth/1e3, color=GOLD, linewidth=1.8,
             label="Luot truy cap web (TB 14 ngay)", alpha=0.9, zorder=3)
    ax1b.plot(rev_smooth.index, rev_smooth/1e6, color=ACCENT, linewidth=1.8,
              label="Doanh thu (TB 14 ngay)", alpha=0.9)

    ax1.set_ylabel("Luot truy cap (nghin)", fontsize=9, color=GOLD)
    ax1b.set_ylabel("Doanh thu (Trieu VND)", fontsize=9, color=ACCENT)
    ax1.tick_params(axis="y", colors=GOLD, labelsize=7)
    ax1b.tick_params(axis="y", colors=ACCENT, labelsize=7)
    ax1.spines[["top"]].set_visible(False)
    ax1b.spines[["top"]].set_visible(False)


    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    ax1.set_facecolor(BG)
    ax1.set_title("Web traffic & Doanh thu cùng nhịp mùa vụ (2 nam gan nhat)",
                  fontsize=11, fontweight="bold", pad=8)
    ax1.tick_params(axis="x", labelsize=7, rotation=20)

    # Scatter: sessions(t-2) vs revenue(t)
    lag_opt = max(0, peak_lag)
    x_lag = wt_aligned.shift(lag_opt).dropna()
    y_rev = rev_aligned.reindex(x_lag.index).dropna()
    x_lag = x_lag.reindex(y_rev.index)

    ax2.scatter(x_lag/1e3, y_rev/1e6, alpha=0.2, s=10, color=ACCENT)
    z = np.polyfit(x_lag, y_rev, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_lag.min(), x_lag.max(), 100)
    ax2.plot(x_line/1e3, p(x_line)/1e6, color=RED, linewidth=2)
    _style_ax(ax2,
              title=f"Phân tán: Lượt truy cập(t−{lag_opt}) vs Doanh thu(t)",
              xlabel="Lượt truy cập web (nghìn)", ylabel="Doanh thu (Triệu VNĐ)")

    fig.suptitle("Lượt truy cập web KHÔNG dẫn trước doanh thu (CCF phẳng ~0.32) — chỉ tương quan mùa vụ chung",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "viz7_web_traffic_ccf.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def viz8_cointegration_anomaly(sales: pd.DataFrame, out_dir: str):
    """
    Viz 8: Revenue và COGS đồng liên kết dài hạn — phát hiện giai đoạn chi phí đội bất thường
    Chỉ dùng sales.csv (Revenue + COGS)
    """
    rev  = sales["Revenue"].dropna()
    cogs = sales["COGS"].dropna()
    common = rev.index.intersection(cogs.index)
    rev  = rev.loc[common]
    cogs = cogs.loc[common]

    # Kiểm định đồng liên kết Engle-Granger
    score, pvalue, _ = coint(rev, cogs)

    # Spread = COGS - beta*Revenue (OLS residual)
    beta = np.cov(cogs, rev)[0, 1] / np.var(rev)
    spread = cogs - beta * rev
    spread_roll_mean = spread.rolling(90).mean()
    spread_roll_std  = spread.rolling(90).std()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("white")

    # Panel 1: Revenue vs COGS
    ax1.plot(rev.index,  rev/1e6,  color=ACCENT, linewidth=1.0, label="Doanh thu", alpha=0.9)
    ax1.plot(cogs.index, cogs/1e6, color=RED,    linewidth=1.0, label="Giá vốn", alpha=0.9)
    _style_ax(ax1, title="Biểu đồ xu hướng dài hạn giữa doanh thu và giá vốn 2012-2022",
              ylabel="Triệu VNĐ")
    ax1.legend(fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    # Panel 2: Spread với dải bất thường (Anomaly bands)
    ax2.plot(spread.index, spread/1e6, color="#888888", linewidth=0.7, alpha=0.7)
    ax2.plot(spread_roll_mean.index, spread_roll_mean/1e6, color=DARK, linewidth=1.5, label="Trung bình trượt 90 ngày")
    upper = (spread_roll_mean + 2*spread_roll_std)
    lower = (spread_roll_mean - 2*spread_roll_std)
    ax2.fill_between(spread.index, lower/1e6, upper/1e6, alpha=0.15, color=GOLD, label="±2 Độ lệch chuẩn")

    # Đánh dấu các điểm bất thường (spread > +2σ)
    anomaly = spread > upper
    ax2.scatter(spread.index[anomaly], spread[anomaly]/1e6, color=RED, s=8, zorder=5,
                label=f"Chi phí đội bất thường ({anomaly.sum()} ngày)")
    
    _style_ax(ax2, title="Biểu đồ chênh lệch giữa doanh thu và giá vốn 2012-2022",
              xlabel="Thời gian", ylabel="Chênh lệch (Triệu VNĐ)")
    ax2.legend(fontsize=8)
    ax2.axhline(0, color="#999999", linewidth=0.8)

    fig.suptitle("Các biểu đồ xu hướng và biến động của chênh lệch chi phí giữa doanh thu và giá vốn 2012-2022",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "viz8_cointegration_anomaly.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def run_diagnostic(sales: pd.DataFrame, clean_revenue: pd.Series, out_dir: str,
                   inventory_path: str, promotions_path: str, web_traffic_path: str):
    """Run tất cả 4 Diagnostic visualizations."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    print("  [Diagnostic] Viz 5: Denoising + stockout...")
    paths.append(viz5_denoising_stockout(sales, inventory_path, clean_revenue, out_dir))
    print("  [Diagnostic] Viz 6: Promotion intervention...")
    paths.append(viz6_promotion_intervention(sales, promotions_path, out_dir))
    print("  [Diagnostic] Viz 7: Web traffic CCF...")
    paths.append(viz7_web_traffic_ccf(sales, web_traffic_path, out_dir))
    print("  [Diagnostic] Viz 8: Cointegration anomaly...")
    paths.append(viz8_cointegration_anomaly(sales, out_dir))
    return paths
