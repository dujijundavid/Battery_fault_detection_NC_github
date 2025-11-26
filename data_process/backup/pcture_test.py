# -*- coding: utf-8 -*-
"""
batch_plot_with_diagnostics.py
按“根目录/车辆编号/若干parquet”批量绘制每车【电压-电流-温度】三联图，并输出“断线原因诊断报告”。

断线的典型原因在本脚本中都会统计：
1) 时间本身不连续：相邻点时间间隔 > GAP_BREAK_SEC（默认 15 分钟）
2) 原始列存在 NaN：max/min单体电压、电流、温度等为 NaN
3) 平均单体电压为 NaN：
   - 总电压 NaN
   - 串数 NaN
   - 串数=0（除零问题）
"""

# ========= 需要你修改的参数 =========
INPUT_ROOT   = r"C:\Users\YIFSHEN\Documents\01_InputRawData\normal_0013"        # 根目录：子文件夹=车辆编号，内含 *.parquet
OUTPUT_DIR   = r"C:\Users\YIFSHEN\Documents\01_InputRawData\normal_0013"        # 图片输出目录
REPORT_DIR   = r"C:\Users\YIFSHEN\Documents\01_InputRawData\normal_0013"     # 诊断CSV输出目录（每车一份 + 汇总一份）
DAYS         = 7                             # 窗口长度（天），取最后 DAYS 天
TIMEZONE     = "Europe/Berlin"                # 图上显示的时区
DOWNSAMPLE   = 1                              # 下采样步长（>=1）
PNG_DPI      = 300                            # 图片清晰度
GAP_BREAK_SEC = 15 * 60                       # “大间隔阈值”（秒），用于判定时间不连续
TOPK_GAPS     = 5                             # 报告里列出最大的前K个时间空档
# ===================================

# —— 固定列名（与你之前的数据一致）——
TIME_COL       = "time"
TOTAL_VOLT_COL = "bms_total_voltage"
CURRENT_COL    = "bms_total_current"
VMAX_COL       = "bms_volt_max_value"
VMIN_COL       = "bms_volt_min_value"
TMAX_COL       = "bms_temp_max_value"
TMIN_COL       = "bms_temp_min_value"
CELL_CNT_COL   = "bms_tba_cells_1"

# =============== 实现部分（无需改） ===============
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from tqdm import tqdm

REQUIRED_COLS = [TIME_COL, TOTAL_VOLT_COL, CURRENT_COL, VMAX_COL, VMIN_COL, TMAX_COL, TMIN_COL, CELL_CNT_COL]

def parse_time_series(ts: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(ts):
        return pd.to_datetime(ts, unit="ms", utc=True)
    return pd.to_datetime(ts, utc=True, errors="coerce")

def read_parquet_cols(p: Path, cols=None) -> pd.DataFrame:
    try:
        return pd.read_parquet(p, columns=cols)
    except TypeError:
        return pd.read_parquet(p)
    except Exception as e:
        raise RuntimeError(f"读取 {p} 失败：{e}")

def find_car_dirs(root: Path):
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and any(sub.glob("*.parquet")):
            yield sub

def get_car_max_ts(car_dir: Path):
    max_ts = None
    for fp in sorted(car_dir.glob("*.parquet")):
        try:
            df = read_parquet_cols(fp, cols=[TIME_COL])
        except Exception as e:
            warnings.warn(str(e)); continue
        if TIME_COL not in df.columns: continue
        ts = parse_time_series(df[TIME_COL])
        if ts.notna().any():
            cur_max = ts.max()
            if (max_ts is None) or (cur_max > max_ts):
                max_ts = cur_max
    return max_ts

def load_car_last_month_df(car_dir: Path, cutoff_utc) -> pd.DataFrame | None:
    frames = []
    for fp in sorted(car_dir.glob("*.parquet")):
        try:
            df = read_parquet_cols(fp, cols=REQUIRED_COLS)
        except Exception as e:
            warnings.warn(str(e)); continue
        miss = [c for c in REQUIRED_COLS if c not in df.columns]
        if miss:
            warnings.warn(f"{fp} 缺列 {miss}，跳过"); continue
        ts = parse_time_series(df[TIME_COL])
        m = ts >= cutoff_utc
        if not m.any(): continue
        sub = df.loc[m, REQUIRED_COLS].copy()
        sub["_ts_utc"] = ts.loc[m].values
        frames.append(sub)
    if not frames: return None
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["_ts_utc"])
    if out.empty: return None
    out = out.sort_values("_ts_utc").reset_index(drop=True)
    return out

def diag_one_car(df: pd.DataFrame, car_id: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """返回 绘图所需df、诊断汇总dict、TopK gap表"""
    # 数值化
    total_v = pd.to_numeric(df[TOTAL_VOLT_COL], errors="coerce")
    cells   = pd.to_numeric(df[CELL_CNT_COL],   errors="coerce")
    v_max   = pd.to_numeric(df[VMAX_COL],       errors="coerce")
    v_min   = pd.to_numeric(df[VMIN_COL],       errors="coerce")
    cur     = pd.to_numeric(df[CURRENT_COL],    errors="coerce")
    t_max   = pd.to_numeric(df[TMAX_COL],       errors="coerce")
    t_min   = pd.to_numeric(df[TMIN_COL],       errors="coerce")

    # 平均单体电压（不做稳定化，先看“真实原因”）
    v_avg = total_v / cells.replace(0, np.nan)

    # 时间间隔
    ts = df["_ts_utc"].astype("int64") // 10**9
    dt = ts.diff().fillna(0).astype(int)
    gap_mask = dt > GAP_BREAK_SEC
    gap_idx = np.where(gap_mask.values)[0]
    gaps = []
    for i in gap_idx:
        if i == 0: continue
        gaps.append({
            "car_id": car_id,
            "gap_start_utc": df["_ts_utc"].iloc[i-1],
            "gap_end_utc":   df["_ts_utc"].iloc[i],
            "gap_seconds":   int(dt.iloc[i]),
            "gap_hours":     float(dt.iloc[i]) / 3600.0
        })
    gap_df = pd.DataFrame(gaps).sort_values("gap_seconds", ascending=False).head(TOPK_GAPS)

    # 缺失统计
    n = len(df)
    def miss(s): return int(s.isna().sum())
    stats = {
        "car_id": car_id,
        "n_points": n,
        # 原始列NaN数
        "nan_total_voltage": miss(total_v),
        "nan_cell_count":    miss(cells),
        "nan_vmax":          miss(v_max),
        "nan_vmin":          miss(v_min),
        "nan_current":       miss(cur),
        "nan_tmax":          miss(t_max),
        "nan_tmin":          miss(t_min),
        # 串数为0
        "zero_cell_count":   int((cells == 0).sum(skipna=True)) if hasattr(cells, "sum") else 0,
        # 平均单体电压NaN分解
        "nan_vavg_total":    miss(v_avg),
        "nan_vavg_due_totalV": int(((total_v.isna()) & (~cells.isna()) & (cells != 0)).sum()),
        "nan_vavg_due_cellsNA": int((cells.isna() & (~total_v.isna())).sum()),
        "nan_vavg_due_cells0":  int(((cells == 0) & (~total_v.isna())).sum()),
        # 时间大间隔
        "n_gaps_gt_thresh": len(gap_idx),
        "max_gap_seconds":  int(gap_df["gap_seconds"].max()) if not gap_df.empty else 0,
    }

    # 综合结论（粗判）
    cause_scores = {
        "Time gaps": stats["n_gaps_gt_thresh"],
        "Orig NaN (V/I/T)": stats["nan_vmax"] + stats["nan_vmin"] + stats["nan_current"] + stats["nan_tmax"] + stats["nan_tmin"],
        "AvgV NaN (total/cells)": stats["nan_vavg_total"],
    }
    primary_cause = max(cause_scores, key=cause_scores.get) if n > 0 else "N/A"
    stats["primary_cause"] = primary_cause

    # 返回绘图df（为了可视化“真实断线”，我们把两类断点置 NaN：时间大空档 + 任一曲线NaN）
    plot_df = pd.DataFrame({
        "_ts_utc": df["_ts_utc"],
        "v_max": v_max, "v_min": v_min, "v_avg": v_avg,
        "current": cur, "t_max": t_max, "t_min": t_min
    })
    # 在“时间大空档”处打断
    for c in ["v_max", "v_min", "v_avg", "current", "t_max", "t_min"]:
        s = plot_df[c].copy()
        s.loc[gap_mask] = np.nan
        plot_df[c] = s

    return plot_df, stats, gap_df

def plot_one_car(plot_df: pd.DataFrame, car_id: str, out_dir: Path):
    # 时区
    try:
        t_disp = plot_df["_ts_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        t_disp = plot_df["_ts_utc"]

    # 下采样
    step = max(1, int(DOWNSAMPLE))
    idx = np.arange(0, len(plot_df), step)
    t = t_disp.iloc[idx]
    v_max = plot_df["v_max"].iloc[idx]
    v_min = plot_df["v_min"].iloc[idx]
    v_avg = plot_df["v_avg"].iloc[idx]
    cur   = plot_df["current"].iloc[idx]
    tmax  = plot_df["t_max"].iloc[idx]
    tmin  = plot_df["t_min"].iloc[idx]

    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    locator = AutoDateLocator(); formatter = AutoDateFormatter(locator)

    ax0 = axes[0]
    ax0.plot(t, v_max, label="Max single V (V)", linewidth=1.1)
    ax0.plot(t, v_min, label="Min single V (V)", linewidth=1.1)
    ax0.plot(t, v_avg, label="Avg single V = total/cells (V)", linewidth=1.0, linestyle="--")
    ax0.set_ylabel("Voltage (V)"); ax0.legend(loc="best"); ax0.grid(True, alpha=0.3)
    ax0.xaxis.set_major_locator(locator); ax0.xaxis.set_major_formatter(formatter)

    ax1 = axes[1]
    ax1.plot(t, cur, label="Current (A)", linewidth=1.0)
    ax1.axhline(0, color="k", linewidth=0.8, alpha=0.4)
    ax1.set_ylabel("Current (A)"); ax1.legend(loc="best"); ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(locator); ax1.xaxis.set_major_formatter(formatter)

    ax2 = axes[2]
    ax2.plot(t, tmax, label="Max Temp (°C)", linewidth=1.0)
    ax2.plot(t, tmin, label="Min Temp (°C)", linewidth=1.0)
    ax2.set_ylabel("Temperature (°C)"); ax2.set_xlabel(f"Time ({TIMEZONE})")
    ax2.legend(loc="best"); ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(locator); ax2.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    start_local = str(t.iloc[0])[:19]; end_local = str(t.iloc[-1])[:19]
    fig.suptitle(f"{car_id} | last {DAYS} days | {start_local} → {end_local}", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{car_id}_last{DAYS}d_vit.png"
    fig.savefig(out_png, dpi=PNG_DPI); plt.close(fig)
    print(f"✅ 图已保存：{out_png}")

def main():
    root = Path(INPUT_ROOT)
    plot_out = Path(OUTPUT_DIR); rep_out = Path(REPORT_DIR)
    if not root.exists():
        raise FileNotFoundError(f"输入根目录不存在：{root}")
    plot_out.mkdir(parents=True, exist_ok=True)
    rep_out.mkdir(parents=True, exist_ok=True)

    car_dirs = list(find_car_dirs(root))
    if not car_dirs:
        raise RuntimeError("未找到任何含 parquet 的车辆子文件夹。")

    all_stats = []
    print(f"共 {len(car_dirs)} 个车辆文件夹，开始诊断（窗口 {DAYS} 天，间隔阈值 {GAP_BREAK_SEC}s）…")
    for car_dir in tqdm(car_dirs, desc="Processing cars"):
        car_id = car_dir.name
        max_ts = get_car_max_ts(car_dir)
        if max_ts is None or pd.isna(max_ts):
            warnings.warn(f"{car_id}: 未找到有效时间戳，跳过"); continue
        cutoff = max_ts - pd.Timedelta(days=max(1, DAYS))
        df = load_car_last_month_df(car_dir, cutoff_utc=cutoff)
        if df is None or df.empty:
            warnings.warn(f"{car_id}: 过滤后为空（max_ts={max_ts}）"); continue

        # 诊断与绘图
        plot_df, stats, gap_df = diag_one_car(df, car_id)
        all_stats.append(stats)
        # 保存单车gap明细
        if not gap_df.empty:
            gap_csv = rep_out / f"{car_id}_gaps_top{TOPK_GAPS}.csv"
            gap_df.to_csv(gap_csv, index=False)
        # 保存图
        try:
            plot_one_car(plot_df, car_id, plot_out)
        except Exception as e:
            warnings.warn(f"{car_id}: 绘图失败：{e}")

        # 保存单车统计
        car_csv = rep_out / f"{car_id}_diagnostics.csv"
        pd.DataFrame([stats]).to_csv(car_csv, index=False)

    # 汇总表
    if all_stats:
        summary = pd.DataFrame(all_stats)
        # 加几个比例列（便于快速判断）
        n = summary["n_points"].replace(0, np.nan)
        for k in ["nan_total_voltage","nan_cell_count","nan_vmax","nan_vmin","nan_current","nan_tmax","nan_tmin","nan_vavg_total"]:
            summary[k+"_rate"] = (summary[k] / n).round(4)
        summary.to_csv(rep_out / "diagnostics_summary.csv", index=False)
        print(f"📄 诊断汇总已保存：{rep_out / 'diagnostics_summary.csv'}")

if __name__ == "__main__":
    main()
