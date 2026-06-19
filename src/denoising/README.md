# Denoising Module

**Buoc 1 — Khu nhieu muc tieu: tao `Clean_Revenue` va `Clean_COGS`.**

## Files

| File | Mo ta |
|------|-------|
| `denoise.py` | Ham xu ly nhieu chinh |
| `__init__.py` | Export: `denoise_target` |

## Ham chinh: `denoise_target(df, stockout_flag_series)`

Xu ly 2 buoc:

### 1. Impute ngay Stockout
- Join `sales.csv` voi `inventory.csv` qua `snapshot_date` (cuoi thang)
- Cac ngay co `stockout_flag = 1` -> doanh thu thap gia tao (khong phai nhu cau giam)
- Impute bang rolling mean 7 ngay trung tam (centered)

### 2. Cap outlier khong lap lai
- Phat hien spike > 3 sigma trong cua so 30 ngay trung tam
- Kiem tra tinh lap lai: spike co xuat hien >= 70% nam lich su khong?
  - Neu co: **giu nguyen** (vi day la spike mua vu thuc su, vd: cuoi thang, Tet)
  - Neu khong: **cap xuong** nguong 3 sigma (nhieu ngau nhien)

## Logic recap
```
Raw Revenue
  --> Impute stockout days (rolling mean 7d)
  --> Cap non-recurring spikes (>3sigma, recurrence <70%)
  --> Clean_Revenue (dau vao cho Prophet)
```

## Input/Output
- **Input:** `df` (DataFrame voi cot Revenue, COGS), `stockout_flag_series` (tu inventory.csv)
- **Output:** `df` co them cot `Clean_Revenue`, `Clean_COGS`
