"""
src/analysis/descriptive.py — Phân tích Mô tả (Descriptive)
"What happened?"

Viz 1: Revenue time series + rolling volatility band
Viz 2: Monthly profit margin heatmap (year × month)
Viz 3: STL Decomposition 3-panel
Viz 4: Revenue by product category (top categories)
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
import seaborn as sns
from statsmodels.tsa.seasonal import STL

# ── Style ─────────────────────────────────────────────────
PALETTE = ["#2EC4B6", "#E71D36", "#FF9F1C", "#011627", "#FDFFFC"]
ACCENT  = "#2EC4B6"
RED     = "#E71D36"
GOLD    = "#FF9F1C"
BG      = "#F8F9FA"

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def viz1_revenue_trend(sales: pd.DataFrame, out_dir: str):
    """
    Viz 1: Mean Daily Revenue Trend & Volatility (2012-2022)
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")

    # Convert to millions for plotting
    rev = sales["Revenue"].copy() / 1e6
    
    # Resample to monthly mean and std of daily revenue
    monthly_mean = rev.resample('ME').mean()
    monthly_std = rev.resample('ME').std()

    # Plot mean daily revenue
    ax.plot(monthly_mean.index, monthly_mean, color="blue", label="Doanh thu bình quân ngày (Triệu VNĐ)")
    
    # Plot +/- 1 Std Dev
    ax.fill_between(monthly_mean.index, monthly_mean - monthly_std, monthly_mean + monthly_std,
                     alpha=0.2, color="blue", label="±1 Độ lệch chuẩn (Biến động)")

    # Linear trend
    x = np.arange(len(monthly_mean))
    mask = ~monthly_mean.isna()
    z = np.polyfit(x[mask], monthly_mean[mask], 1)
    p = np.poly1d(z)
    ax.plot(monthly_mean.index, p(x), color="red", linestyle="--", label="Xu hướng tuyến tính")

    ax.set_title("Biểu đồ xu hướng doanh thu trong khoảng thời gian 2012-2022", fontsize=12)
    ax.set_ylabel("Doanh thu ngày (Triệu VNĐ)", fontsize=10)
    ax.set_xlabel("Thời gian", fontsize=10)
    
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "viz1_revenue_trend.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path



def viz2_profit_margin_heatmap(sales: pd.DataFrame, out_dir: str):
    """
    Viz 2: Biên lợi nhuận gộp dao động theo mùa — phát hiện tháng 8 margin âm bất thường
    """
    df = sales.copy()
    df["margin"] = (df["Revenue"] - df["COGS"]) / df["Revenue"] * 100
    df["year"]   = df.index.year
    df["month"]  = df.index.month

    pivot = df.groupby(["year", "month"])["margin"].mean().unstack()
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")

    # Use explicit vmin/vmax to handle extreme negative margins (Aug = -42%)
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                vmin=-45, vmax=25, center=0, linewidths=0.5,
                annot_kws={"size": 9, "fontweight": "bold"}, ax=ax,
                cbar_kws={"label": "Biên lợi nhuận gộp (%)", "shrink": 0.8})

    ax.set_title("Biên lợi nhuận gộp (%) theo năm và tháng: Tháng 8 thường âm, Q1 cao nhất",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Tháng", fontsize=10)
    ax.set_ylabel("Năm", fontsize=10)
    ax.tick_params(labelsize=9)

    # Highlight anomalous cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val) and val < 0:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                             edgecolor=RED, linewidth=2.5))

    plt.tight_layout()
    path = os.path.join(out_dir, "viz2_profit_margin_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def viz3_stl_decomposition(sales: pd.DataFrame, out_dir: str):
    """
    Viz 3: Phân rã STL — Mùa vụ chiếm ~35% biến động
    """
    rev_daily = sales["Revenue"].resample("D").mean().interpolate()
    stl = STL(rev_daily, period=365, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Phân rã STL: Mùa vụ & Xu hướng chiếm 65% tổng biến động doanh thu",
                 fontsize=12, fontweight="bold", y=1.01)

    components = [
        (rev_daily,     "Observed (Doanh thu gốc)",   ACCENT),
        (res.trend,     "Trend (Xu hướng dài hạn)",   "#011627"),
        (res.seasonal,  "Seasonal (Mùa vụ hàng năm)", GOLD),
        (res.resid,     "Residual (Nhiễu / Bất thường)", RED),
    ]
    for ax, (data, label, color) in zip(axes, components):
        ax.plot(data.index, data.values, color=color, linewidth=0.8)
        ax.set_ylabel(label, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor(BG)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

    axes[-1].set_xlabel("Thời gian", fontsize=9)

    # Annotate variance share
    var_seasonal = float(np.var(res.seasonal))
    var_total    = float(np.var(rev_daily))
    pct = var_seasonal / var_total * 100
    axes[2].annotate(f"Biến động mùa vụ: {pct:.0f}% tổng",
                     xy=(rev_daily.index[100], float(res.seasonal.iloc[100])),
                     fontsize=8, color=GOLD, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "viz3_stl_decomposition.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def viz4_revenue_by_category(order_items_path: str, products_path: str,
                              orders_path: str, out_dir: str):
    """
    Viz 4: Top categories chiếm 72% doanh thu — JOIN order_items + products + orders
    """
    oi = pd.read_csv(order_items_path)
    pr = pd.read_csv(products_path)[["product_id", "category", "cogs"]]
    od = pd.read_csv(orders_path)[["order_id", "order_date"]]
    od["order_date"] = pd.to_datetime(od["order_date"])
    od["year"] = od["order_date"].dt.year

    merged = oi.merge(pr, on="product_id").merge(od, on="order_id")
    merged["line_revenue"] = merged["quantity"] * merged["unit_price"] - merged["discount_amount"]
    merged["line_cogs"]    = merged["quantity"] * merged["cogs"]

    cat_rev = merged.groupby("category")["line_revenue"].sum().sort_values(ascending=False)
    total   = cat_rev.sum()
    cumpct  = cat_rev.cumsum() / total * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("white")

    # Pie chart
    DARK = "#011627"
    colors = [ACCENT, GOLD, RED, DARK, "#AAAAAA"][:len(cat_rev)]
    
    wedges, texts, autotexts = ax1.pie(cat_rev.values, labels=None, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 9, 'weight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}, pctdistance=0.7)
            
    # Di chuyển % ra ngoài nếu tỉ trọng < 5%
    for i, a in enumerate(autotexts):
        if (cat_rev.values[i] / total * 100) < 5.0:
            theta = np.deg2rad((wedges[i].theta1 + wedges[i].theta2) / 2)
            a.set_position((1.2 * np.cos(theta), 1.2 * np.sin(theta)))
            # Thêm đường nối mờ để dễ nhìn
            ax1.annotate("", xy=(0.95 * np.cos(theta), 0.95 * np.sin(theta)),
                         xytext=(1.1 * np.cos(theta), 1.1 * np.sin(theta)),
                         arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8))

    ax1.legend(wedges, cat_rev.index, title="Danh mục", loc="center left", bbox_to_anchor=(0.95, 0.5), fontsize=8)
    ax1.set_title("Biểu đồ tròn tổng tỉ trọng doanh thu theo từng danh mục 2012–2022", fontsize=11, fontweight="bold", pad=8)

    # Yearly trend by all categories (Stacked Area Chart)
    all_cats = cat_rev.index.tolist()
    yearly = merged.groupby(["year", "category"])["line_revenue"].sum().unstack().fillna(0)
    
    plot_data = []
    plot_labels = []
    plot_colors = []
    
    for cat, color in zip(all_cats, colors):
        if cat in yearly.columns:
            plot_data.append(yearly[cat] / 1e9)
            plot_labels.append(cat)
            plot_colors.append(color)

    ax2.stackplot(yearly.index, plot_data, labels=plot_labels, colors=plot_colors, alpha=0.85)

    _style_ax(ax2, title="Biểu đồ miền xếp chồng cấu trúc doanh thu theo danh mục 2012-2022",
              xlabel="Năm", ylabel="Doanh thu (Tỷ VNĐ)")
    ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle("Phân tích doanh thu theo danh mục sản phẩm (2012–2022)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "viz4_revenue_by_category.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def run_descriptive(sales: pd.DataFrame, out_dir: str,
                    order_items_path: str, products_path: str, orders_path: str):
    """Run tất cả 4 Descriptive visualizations."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    print("  [Descriptive] Viz 1: Revenue trend...")
    paths.append(viz1_revenue_trend(sales, out_dir))
    print("  [Descriptive] Viz 2: Profit margin heatmap...")
    paths.append(viz2_profit_margin_heatmap(sales, out_dir))
    print("  [Descriptive] Viz 3: STL decomposition...")
    paths.append(viz3_stl_decomposition(sales, out_dir))
    print("  [Descriptive] Viz 4: Revenue by category...")
    paths.append(viz4_revenue_by_category(order_items_path, products_path, orders_path, out_dir))
    return paths
