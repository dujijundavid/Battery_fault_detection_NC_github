# -*- coding: utf-8 -*-
"""
基于你现有的列清洗逻辑，统计“所有充电段（不做过滤）”在 SOC 区间的出现次数
- 充电段判定：bit_charging_state == 'CHARGING_IN_PARKING' 且 current > 0.01
- 切段规则：相邻时间差 > 300 s（5 分钟）断段
- 不做任何段过滤（不看起末 SOC、不看长度等）
- 目录结构：IN_ROOT/车ID子文件夹/*.parquet（可多个）

输出：
  1) soc_segment_distribution.csv / .png
  2) segments_soc_coverage.csv（每个段的起止时间、SOC范围、覆盖的SOC区间等）
"""

import os, glob, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============== 路径与参数（按需修改） ===============
IN_ROOT  = r"C:\Users\YIFSHEN\Documents\01_InputRawData\3000_distinct_sample_v1_cleaned_new"
OUT_DIR  = r"C:\Users\YIFSHEN\Documents\01_InputRawData\3000_distinct_sample_v1_cleaned_new"
PARQUET_GLOB = "*.parquet"

# 与你现有代码保持一致的清洗/判定口径
SAMPLE_STEP_SEC = 30      # 仅用于首个 diff 填充值
GAP_BREAK_SEC   = 300     # 5 分钟断段
CURR_THRESH     = 0.01    # 认为“正在充电”的电流阈值（安）
CHG_STATUS_VAL  = "CHARGING_IN_PARKING"  # 充电状态字符串值

# SOC 统计分箱（百分比）
SOC_MIN, SOC_MAX = 0, 100
BIN_STEP         = 5
ANNOTATE_BARS    = False
# =====================================================


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_parquet_files(root: str):
    """遍历 root 下的一级子目录，收集每个子目录内的所有 .parquet 文件。"""
    subdirs = [d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d)]
    pairs = []  # (subdir_name, parquet_path)
    for sd in subdirs:
        sd_name = os.path.basename(sd.rstrip(r"\/"))
        for pq in glob.glob(os.path.join(sd, PARQUET_GLOB)):
            pairs.append((sd_name, pq))
    return pairs


def read_and_clean(parquet_path: str) -> pd.DataFrame:
    """
    读取 parquet，并按你给的 clean_and_select_columns 逻辑清洗/重命名。
      输出列：
        time(秒, int)、mileage、status(str)、volt(平均单体电压)、current、soc(0..100)、
        max_single_volt、min_single_volt、max_temp、min_temp、cells
    """
    # 只读必要列以减少 IO
    need_cols = [
        'time', 'odo', 'bit_charging_state', 'bms_total_voltage', 'bms_total_current', 'bms_soc',
        'bms_volt_max_value', 'bms_volt_min_value', 'bms_temp_max_value', 'bms_temp_min_value',
        'bms_tba_cells_1'
    ]
    raw = pd.read_parquet(parquet_path, columns=need_cols, engine='pyarrow')

    # 列检查
    miss = [c for c in need_cols if c not in raw.columns]
    if miss:
        raise KeyError(f"缺失必须字段: {miss}")

    # 重命名
    tmp = raw.rename(columns={
        'time': 'time',
        'odo': 'mileage',
        'bit_charging_state': 'status',
        'bms_total_voltage': 'volt',
        'bms_total_current': 'current',
        'bms_soc': 'soc',
        'bms_volt_max_value': 'max_single_volt',
        'bms_volt_min_value': 'min_single_volt',
        'bms_temp_max_value': 'max_temp',
        'bms_temp_min_value': 'min_temp',
        'bms_tba_cells_1': 'cells',
    }).copy()

    # time：毫秒->秒（向下取整）
    tmp['time'] = pd.to_numeric(tmp['time'], errors='coerce').fillna(0).astype(np.int64) // 1000

    # 数值列
    num_cols = ['mileage', 'volt', 'current', 'soc',
                'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp', 'cells']
    for c in num_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors='coerce')

    # 状态为字符串（去空白）
    tmp['status'] = tmp['status'].astype(str).str.strip()

    # 宽松清洗（与你一致）
    tmp.loc[~tmp['soc'].between(0, 100), 'soc'] = np.nan
    tmp.loc[tmp['volt'] <= 0, 'volt'] = np.nan
    tmp.loc[tmp['max_single_volt'] <= 0, 'max_single_volt'] = np.nan
    tmp.loc[tmp['min_single_volt'] <= 0, 'min_single_volt'] = np.nan
    tmp.loc[tmp['max_temp'] < -100, 'max_temp'] = np.nan
    tmp.loc[tmp['min_temp'] < -100, 'min_temp'] = np.nan

    # 平均单体电压：总电压 / 电芯数
    tmp['cells'] = tmp['cells'].where(tmp['cells'] > 0, np.nan)
    tmp['volt']  = tmp['volt'] / tmp['cells']

    # 丢弃关键列 NaN
    tmp = tmp.dropna(subset=[
        'time', 'volt', 'current', 'soc',
        'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp'
    ])

    # 排序 + 去重时间
    tmp = tmp.sort_values('time').drop_duplicates(subset='time', keep='first').reset_index(drop=True)
    return tmp


def pick_charging_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    充电状态筛选（与原代码一致）：
    status == 'CHARGING_IN_PARKING' 且 current > 0.01
    """
    m = (df['status'] == CHG_STATUS_VAL) & (df['current'] > CURR_THRESH)
    return df.loc[m].copy()


def split_by_time_gap(df: pd.DataFrame) -> list:
    """
    按 5 分钟断点切段（不做任何额外过滤）
    返回 list[pd.DataFrame] 每个元素是一段
    """
    if df.empty:
        return []
    # time 已是秒
    time_diff = df['time'].diff().fillna(SAMPLE_STEP_SEC)
    session_id = (time_diff > GAP_BREAK_SEC).cumsum()
    segs = [g.reset_index(drop=True) for _, g in df.groupby(session_id, sort=True)]
    return segs


def main():
    out_dir = Path(OUT_DIR); safe_mkdir(out_dir)

    # SOC 分箱
    edges = np.arange(SOC_MIN, SOC_MAX + BIN_STEP, BIN_STEP, dtype=float)
    if edges[-1] > SOC_MAX:
        edges[-1] = SOC_MAX
    nbins  = len(edges) - 1
    labels = [f"{int(l)}–{int(r)}" if (l.is_integer() and r.is_integer())
              else f"{l:.1f}–{r:.1f}" for l, r in zip(edges[:-1], edges[1:])]

    bin_counts = np.zeros(nbins, dtype=int)
    seg_rows   = []

    pairs = list_parquet_files(IN_ROOT)
    if not pairs:
        print(f"[警告] 在 {IN_ROOT} 下未发现车辆子文件夹或 parquet 文件")
        return

    total_segments = 0
    used_segments  = 0

    # 为了便于定位问题，记录 status 的值分布
    status_value_counter = {}

    pbar = tqdm(pairs, desc="提取与统计")
    for subfolder_name, pq_path in pbar:
        try:
            df = read_and_clean(pq_path)
        except Exception as e:
            print(f" 读取失败 [{subfolder_name}] {pq_path}: {e}")
            continue

        # 统计 status 出现频次，便于排查是否没有 CHARGING_IN_PARKING
        vals, cnts = np.unique(df['status'].astype(str).values, return_counts=True)
        for v, c in zip(vals, cnts):
            status_value_counter[v] = status_value_counter.get(v, 0) + int(c)

        # 充电状态子集
        cand = pick_charging_df(df)
        if cand.empty:
            continue

        # 切段（不做过滤）
        segs = split_by_time_gap(cand)
        if not segs:
            continue

        # 逐段覆盖到哪些 SOC bin
        seg_idx = 0
        for seg in segs:
            seg_idx += 1
            total_segments += 1
            vals_soc = seg['soc'].to_numpy(dtype=float, copy=False)
            idx = np.digitize(vals_soc, bins=edges, right=False) - 1
            idx = idx[(idx >= 0) & (idx < nbins)]
            if idx.size == 0:
                continue
            uniq_bins = np.unique(idx)
            bin_counts[uniq_bins] += 1
            used_segments += 1

            seg_rows.append({
                "vehicle_id": subfolder_name,
                "parquet_file": os.path.basename(pq_path),
                "segment_idx_in_file": seg_idx,
                "start_time": float(seg['time'].iloc[0]),
                "end_time":   float(seg['time'].iloc[-1]),
                "len_points": int(len(seg)),
                "start_soc":  float(seg['soc'].iloc[0]),
                "end_soc":    float(seg['soc'].iloc[-1]),
                "min_soc":    float(np.nanmin(vals_soc)),
                "max_soc":    float(np.nanmax(vals_soc)),
                "covered_bins": ";".join(labels[i] for i in uniq_bins),
            })

    # 输出 1：总体分布
    result = pd.DataFrame({
        "soc_bin_left":  edges[:-1],
        "soc_bin_right": edges[1:],
        "soc_bin_label": labels,
        "segment_count": bin_counts
    })
    csv_path = out_dir / "soc_segment_distribution.csv"
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 画图
    plt.figure(figsize=(14, 6), dpi=150)
    bars = plt.bar(result["soc_bin_label"], result["segment_count"])
    plt.xlabel("SOC 区间（%）")
    plt.ylabel("充电段出现次数")
    plt.title(f"SOC 区间分布（所有充电段，无过滤）\n总切段数：{total_segments}，有效（覆盖到分箱）的段数：{used_segments}")
    plt.xticks(rotation=45, ha="right")
    if ANNOTATE_BARS:
        for r in bars:
            h = r.get_height()
            if h > 0:
                plt.text(r.get_x()+r.get_width()/2, h, f"{int(h)}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig_path = out_dir / "soc_segment_distribution.png"
    plt.savefig(fig_path)
    plt.close()

    # 输出 2：每段摘要
    seg_df = pd.DataFrame(seg_rows, columns=[
        "vehicle_id","parquet_file","segment_idx_in_file",
        "start_time","end_time","len_points",
        "start_soc","end_soc","min_soc","max_soc","covered_bins"
    ])
    seg_csv = out_dir / "segments_soc_coverage.csv"
    seg_df.to_csv(seg_csv, index=False, encoding="utf-8-sig")

    # 附带输出一个 status 值频次，方便排查
    status_stat = (out_dir / "charging_status_value_counts.csv")
    pd.Series(status_value_counter, name="count").sort_values(ascending=False).to_csv(status_stat, header=True, encoding="utf-8-sig")

    print("====== 完成 ======")
    print(f"总切段数：{total_segments}，统计到的有效充电段数：{used_segments}")
    print(f"总体分布：{csv_path}")
    print(f"直方图：{fig_path}")
    print(f"段摘要表：{seg_csv}")
    print(f"status 值分布（排查用）：{status_stat}")


if __name__ == "__main__":
    main()
