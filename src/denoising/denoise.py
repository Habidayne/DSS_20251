"""
src/denoising/denoise.py — Bước 1: Khử nhiễu mục tiêu (Target Denoising).

Mục tiêu: Tạo Clean_Revenue và Clean_COGS bằng cách:
  - Cap spike bất thường (> mean + 3σ trong sliding window 30 ngày),
    nhưng BẢO TỒN spike mang tính hệ thống (end-of-month, lặp hàng năm).

LƯU Ý (PH1, 2026-06-19): Đã GỠ BỎ imputation theo stockout_flag.
Kiểm chứng cho thấy `stockout_flag` tổng hợp theo tháng-SKU gần như hằng số
(~67% mọi tháng, 126/126 tháng > 0.5) → là proxy áp lực tồn kho cấp tháng,
KHÔNG phải sự kiện hết hàng cấp ngày. Dùng nó để vá target chỉ chà phẳng
chuỗi một cách vô căn cứ. Tín hiệu tồn kho dùng được nằm ở `overstock_pct`
(phương sai thật 0.34–0.93) và sẽ vào mô hình dưới dạng feature, không phải
qua imputation.
"""
import pandas as pd
import logging

logger = logging.getLogger("gridbreaker")


def _detect_recurring_spikes(series: pd.Series, threshold: float = 0.7) -> pd.Series:
    """
    Phát hiện spike lặp lại hàng năm (seasonal signature).
    Nếu cùng (month, day) có spike trong >= threshold*tổng số năm → giữ lại.
    """
    df = pd.DataFrame({"val": series, "month": series.index.month, "day": series.index.day, "year": series.index.year})
    # Tính z-score theo năm
    annual_mean = df.groupby("year")["val"].transform("mean")
    annual_std = df.groupby("year")["val"].transform("std")
    df["zscore"] = (df["val"] - annual_mean) / annual_std
    df["is_spike"] = df["zscore"] > 2.0

    # Tỷ lệ năm có spike cho mỗi (month, day)
    spike_rate = df.groupby(["month", "day"])["is_spike"].mean()
    recurring = spike_rate[spike_rate >= threshold].index
    is_recurring = pd.Series(False, index=series.index)
    for m, d in recurring:
        mask = (series.index.month == m) & (series.index.day == d)
        is_recurring[mask] = True
    return is_recurring


def _cap_outliers(series: pd.Series, window: int = 30, n_sigma: float = 3.0) -> pd.Series:
    """Cap spike bất thường, giữ nguyên recurring spikes."""
    recurring = _detect_recurring_spikes(series)
    rolling_mean = series.rolling(window, center=True, min_periods=7).mean()
    rolling_std = series.rolling(window, center=True, min_periods=7).std()
    upper = rolling_mean + n_sigma * rolling_std
    is_outlier = (series > upper) & (~recurring)
    cleaned = series.copy()
    cleaned[is_outlier] = upper[is_outlier]
    n_capped = is_outlier.sum()
    logger.info(f"  Capped {n_capped} non-recurring outlier(s).")
    return cleaned


def denoise_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 1: Tạo Clean_Revenue và Clean_COGS bằng outlier capping.

    Chỉ cap non-recurring spike (giữ recurring seasonal spike). KHÔNG còn
    imputation theo stockout (xem docstring module — PH1).

    Args:
        df: DataFrame với index=Date, columns=['Revenue','COGS']
    Returns:
        df with new columns: Clean_Revenue, Clean_COGS
    """
    logger.info("=" * 50)
    logger.info("BƯỚC 1: TARGET DENOISING (outlier capping, no stockout impute)")
    logger.info("=" * 50)

    result = df.copy()
    for col in ["Revenue", "COGS"]:
        logger.info(f"Processing {col}:")
        cleaned = _cap_outliers(df[col].dropna())
        result[f"Clean_{col}"] = cleaned

    logger.info(f"  Revenue: mean={result['Revenue'].mean():,.0f} → Clean mean={result['Clean_Revenue'].mean():,.0f}")
    logger.info(f"  COGS:    mean={result['COGS'].mean():,.0f} → Clean mean={result['Clean_COGS'].mean():,.0f}")
    return result
