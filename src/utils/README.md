# Utils Module

**Buoc 0 — Tien ich chung cho toan bo pipeline.**

## Files

| File | Mo ta |
|------|-------|
| `helpers.py` | Ham tien ich dung chung |
| `__init__.py` | Export: `setup_logger`, `mape`, `load_sales`, `load_inventory_flags`, `DATA_DIR`, `SEED` |

## Ham chinh

### `setup_logger(log_path)`
Tao logger ghi ca console va file. Mac dinh ghi vao `log.txt` tai project root.

### `load_sales(path)`
Load `sales.csv`, parse dates, set `Date` index, reindex daily, interpolate khoang trong.

### `load_inventory_flags(path)`
Load `inventory.csv`, tong hop `stockout_flag` theo snapshot_date (max theo san pham).
Tra ve Series index theo ngay.

### `mape(y_true, y_pred)`
Tinh Mean Absolute Percentage Error (%).

## Constants
- `SEED = 42` — random seed toan cuc, dam bao reproducibility
- `DATA_DIR` — tu dong phat hien project root tu vi tri file
