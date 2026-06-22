"""
src/prophet_model/train_prophet.py — Bước 2: Tầng 1 cốt lõi với Prophet.

Huấn luyện Prophet trên Clean_Revenue / Clean_COGS.
- Yearly + Weekly seasonality (Fourier)
- Custom holidays: end-of-month spike, Tết Nguyên Đán, Black Friday
- Trend changepoint detection tự động
"""
import pandas as pd
import numpy as np
import logging
from prophet import Prophet

logger = logging.getLogger("gridbreaker")


def _build_holidays() -> pd.DataFrame:
    """Tạo bảng holidays cho Prophet: end-of-month spikes & các ngày lễ đặc biệt."""
    rows = []

    # End-of-month (ngày 28-31 thường có spike doanh thu)
    for year in range(2012, 2025):
        for month in range(1, 13):
            # Last 3 days of month
            if month == 12 and year == 2024:
                continue
            try:
                eom = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
                for delta in range(4):
                    d = eom - pd.Timedelta(days=delta)
                    rows.append({"holiday": "end_of_month", "ds": d})
            except:
                pass

    # First 3 days of month (post-spike recovery)
    for year in range(2012, 2025):
        for month in range(1, 13):
            for day in [1, 2, 3]:
                try:
                    rows.append({"holiday": "start_of_month", "ds": pd.Timestamp(year, month, day)})
                except:
                    pass

    # Tết Nguyên Đán (xấp xỉ, các năm 2012-2024)
    tet_dates = [
        "2012-01-23", "2013-02-10", "2014-01-31", "2015-02-19",
        "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05",
        "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10",
    ]
    for td in tet_dates:
        base = pd.Timestamp(td)
        for delta in range(-2, 8):  # 10 ngày quanh Tết
            rows.append({"holiday": "tet", "ds": base + pd.Timedelta(days=delta)})

    # Khuyến mãi tất định (A5/PH2). Promo căn lịch dương ±1 ngày qua 10 năm
    # (kiểm chứng từ promotions.csv) → encode thành Prophet event để Tầng 1 học
    # trực tiếp thay vì để lại trong residual. (start_doy, duration) lấy từ data.
    # ODD = chỉ chạy năm lẻ (biennial) → Urban Blowout là clearance tháng 8 năm lẻ
    # gây biên lợi nhuận âm; 2023 (lẻ) có, 2024 (chẵn) không.
    PROMO_ALL = {  # name: (start_doy, duration_days) — chạy mọi năm
        "promo_spring":  (77, 30),
        "promo_midyear": (174, 29),
        "promo_fall":    (242, 32),
        "promo_yearend": (322, 45),
    }
    PROMO_ODD = {  # chỉ năm lẻ
        "promo_rural":         (30, 30),
        "promo_urban_blowout": (211, 34),
    }
    for year in range(2012, 2025):
        jan1 = pd.Timestamp(year, 1, 1)
        for name, (doy, dur) in PROMO_ALL.items():
            base = jan1 + pd.Timedelta(days=doy - 1)
            for k in range(dur):
                rows.append({"holiday": name, "ds": base + pd.Timedelta(days=k)})
        if year % 2 == 1:
            for name, (doy, dur) in PROMO_ODD.items():
                base = jan1 + pd.Timedelta(days=doy - 1)
                for k in range(dur):
                    rows.append({"holiday": name, "ds": base + pd.Timedelta(days=k)})

    return pd.DataFrame(rows)


def fit_prophet(series: pd.Series, target_name: str = "Revenue") -> Prophet:
    """
    Huấn luyện Prophet trên chuỗi sạch.
    
    Args:
        series: pd.Series(index=DatetimeIndex, values=Clean_Revenue hoặc Clean_COGS)
        target_name: tên biến (để log)
    Returns:
        model: Prophet đã fit
    """
    logger.info(f"  Fitting Prophet cho {target_name}...")

    train_df = pd.DataFrame({"ds": series.index, "y": series.values})
    holidays = _build_holidays()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=0.15,   # linh hoạt hơn default (0.05)
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_range=0.9,
        # Multiplicative: doanh thu có phương sai tăng theo level (đỉnh 2016 dao
        # động gấp ~3 lần 2013) → biên độ mùa vụ scale theo trend, không cố định.
        # Rolling-origin backtest (prophet_mult_backtest.py) xác nhận thắng ĐỀU
        # 3/3 fold: val2020 ΔMAE−41% (additive R²=−0.18 khi level sụt COVID),
        # val2021 −2.8%, val2022 −5.7%.
        seasonality_mode="multiplicative",
    )
    model.fit(train_df)

    logger.info(f"  Prophet {target_name}: fitted with {len(train_df)} datapoints, {len(holidays)} holiday rows.")
    return model


def predict_prophet(model: Prophet, start: str, end: str) -> pd.DataFrame:
    """
    Dự báo Prophet trên khoảng [start, end].
    
    Returns:
        DataFrame: index=Date, columns=['prophet_pred', 'prophet_trend']
    """
    future = model.make_future_dataframe(
        periods=(pd.Timestamp(end) - model.history["ds"].max()).days,
        freq="D"
    )
    forecast = model.predict(future)
    forecast = forecast.set_index("ds")
    result = forecast.loc[start:end, ["yhat", "trend"]].rename(
        columns={"yhat": "prophet_pred", "trend": "prophet_trend"}
    )
    return result
