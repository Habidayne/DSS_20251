"""
src/utils/helpers.py — Các hàm tiện ích dùng chung cho toàn bộ pipeline.
"""
import logging
import os
import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..")  # project root


def setup_logger(log_path: str = None) -> logging.Logger:
    """Tạo logger ghi ra cả console và file log.txt."""
    if log_path is None:
        log_path = os.path.join(DATA_DIR, "log.txt")

    logger = logging.getLogger("gridbreaker")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", "%Y-%m-%d %H:%M:%S")

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def load_sales(path: str = None) -> pd.DataFrame:
    """
    Tiền xử lý dữ liệu Sales theo 4 bước chuẩn mực:
    1. Data Consolidation: Load và chuẩn hoá trục thời gian.
    2. Data Cleaning: Nội suy dữ liệu khuyết.
    3. Data Reduction: Lọc năm không đủ dữ liệu (Threshold-based filtering).
    4. Data Transformation: (Sẽ thực hiện ở bước feature engineering, ở đây ta trả về dữ liệu chuẩn).
    """
    if path is None:
        path = os.path.join(DATA_DIR, "sales.csv")
    
    # BƯỚC 1: HỢP NHẤT DỮ LIỆU (Data Consolidation)
    # Gom nhóm theo ngày và tạo bộ khung thời gian liên tục
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    full_idx = pd.date_range(df.index.min(), df.index.max())
    df = df.reindex(full_idx)
    
    # BƯỚC 2: LÀM SẠCH DỮ LIỆU (Data Cleaning)
    # Xử lý missing values sinh ra từ bước reindex bằng nội suy tuyến tính theo thời gian
    df["Revenue"] = df["Revenue"].interpolate(method="time")
    df["COGS"] = df["COGS"].interpolate(method="time")
    
    # BƯỚC 3: THU GỌN DỮ LIỆU (Data Reduction)
    # Kỹ thuật: Threshold-based filtering
    # Lý do: Năm 2012 chỉ có 181 ngày, tạo nhiễu cho mô hình Time Series khi tính lag_365.
    # Ta chỉ giữ lại các năm có tối thiểu 360 ngày dữ liệu.
    year_counts = df.groupby(df.index.year).size()
    valid_years = year_counts[year_counts >= 360].index
    
    # Lọc bỏ năm 2012 (và các năm không đủ ngưỡng)
    df = df[df.index.year.isin(valid_years)]
    
    return df


def load_inventory_flags(path: str = None) -> pd.DataFrame:
    """Load inventory.csv, tổng hợp stockout_flag theo ngày (resample từ tháng)."""
    if path is None:
        path = os.path.join(DATA_DIR, "inventory.csv")
    inv = pd.read_csv(path)
    # Tạo date column từ year+month (snapshot ở tháng)
    inv["snapshot_date"] = pd.to_datetime(
        inv["year"].astype(str) + "-" + inv["month"].astype(str) + "-01"
    )
    # Tổng hợp: ngày đầu tháng có ≥ 1 product stockout → flag = 1
    monthly_stockout = (
        inv.groupby("snapshot_date")["stockout_flag"]
        .max()
        .reset_index()
        .rename(columns={"snapshot_date": "Date"})
        .set_index("Date")
    )
    
    # Resample ra daily (ffill cho cả tháng)
    daily_stockout = monthly_stockout.resample("D").ffill()
    
    return daily_stockout
