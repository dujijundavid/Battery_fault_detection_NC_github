# -*- coding: utf-8 -*-
"""
slice_and_save_from_parquet_v6_infer (Databricks 适配版)
用途：模型“推理”所需的数据处理（低内存、严格时间一致性）
说明：在 Databricks 中运行，通过 widgets 配置参数
"""
import os, glob, math, re, shutil
from collections import OrderedDict, defaultdict
from multiprocessing import Pool
from functools import partial
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pyspark.sql import SparkSession

# ===== 固定口径（与原逻辑一致的门槛） =====
SAMPLE_STEP_SEC = 30
GAP_BREAK_SEC = 300
CURR_THRESH = 0.01
MIN_SEG_POINTS = 25
STATUS_VALUE = "CHARGING_IN_PARKING"
USE_SOC_LIMIT = False
SOC_START_LT = 50.0
SOC_END_GT = 65.0
# ===== 30s 轻量规范化参数 =====
STEP = 30 # 目标间隔秒
TOL = 3 # 容差：|dt-STEP|<=3
FILL_LIMIT = 1 # 每个子段最多补 1 帧（针对 ~60s 的缺口）
HARD_GAP = 90 # >=(90±3)s 视为大间断，就地切段
COLS_FEAT = ['volt','current','soc','max_single_volt','min_single_volt','max_temp','min_temp']
REQ_COLS = ['time','time_ms'] + COLS_FEAT
# ===== 必需列映射 =====
COL_MAP = {
    'time': 'time', # UTC毫秒
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
}

# ===== Databricks 环境配置 =====
# 初始化 Spark 会话（Databricks 已内置，此处兼容）
spark = SparkSession.builder.getOrCreate()

# 添加参数 widgets（全部用 text 类型，避免版本兼容问题）
dbutils.widgets.text("raw_folder", "/Workspace/Users/pssx_a_yifshen@corpdir.partner.onmschina.cn/VAE/DATA/normal/", "原始数据根目录")
dbutils.widgets.text("out_folder", "/Workspace/Users/pssx_a_yifshen@corpdir.partner.onmschina.cn/VAE/pkl_normal", "输出目录")
dbutils.widgets.text("window_len", "48", "窗口长度（整数）")  # 改为 text 类型
dbutils.widgets.text("interval", "2", "滑窗步长（整数）")     # 改为 text 类型
dbutils.widgets.text("label", "0", "标签值")
dbutils.widgets.text("end_months_ago", "6", "结束月偏移（整数）")  # 改为 text 类型
dbutils.widgets.text("cover_months", "36", "覆盖月数（整数）")    # 改为 text 类型
dbutils.widgets.text("tz", "UTC", "时区")
dbutils.widgets.text("cst_tz", "Asia/Taipei", "CST展示时区")
dbutils.widgets.text("car_whitelist", "", "车辆白名单（逗号分隔或正则）")
dbutils.widgets.text("max_files", "-1", "最大文件数（整数，-1表示无限制）")  # 改为 text 类型
# dbutils.widgets.text("jobs", str(min(8, os.cpu_count() or 4)), "并行进程数（整数）")  # 改为 text 类型
dbutils.widgets.text("jobs", "32", "并行进程数（整数）") 

# ===== 小工具 =====
def pretty(ts_ms, tz="UTC"):
    """将UTC毫秒时间戳格式化为可读字符串"""
    return pd.to_datetime(int(ts_ms), unit="ms", utc=True).tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

def list_car_dirs(root, car_whitelist=None):
    """列出车辆目录（适配 Databricks 文件系统）"""
    # 处理 DBFS 路径（转换为本地路径）
    if root.startswith("dbfs:/"):
        root = "/dbfs" + root[5:]
    subs = [d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d)]
    if car_whitelist:
        if any(ch in car_whitelist for ch in '*.?|[]()+\\^$'):
            pat = re.compile(car_whitelist)
            subs = [d for d in subs if pat.search(os.path.basename(d))]
        else:
            wl = {s.strip() for s in car_whitelist.split(',') if s.strip()}
            subs = [d for d in subs if os.path.basename(d) in wl]
    return sorted(subs)

def list_parquet_files_of_car(car_dir):
    """列出车辆的 parquet 文件"""
    return sorted(glob.glob(os.path.join(car_dir, '*.parquet')))

def detect_car_anchor_ms(car_dir):
    """读取该车所有parquet的time列最大值(UTC毫秒)为锚点"""
    anchor = None
    for pq in list_parquet_files_of_car(car_dir):
        try:
            # 优先使用 pyarrow 引擎（Databricks 通常已安装）
            s = pd.read_parquet(pq, columns=['time'], engine='pyarrow')['time']
        except Exception:
            s = pd.read_parquet(pq)['time']
        s = pd.to_numeric(s, errors='coerce')
        if s.size:
            mx = s.max(skipna=True)
            if pd.notna(mx):
                mx = int(mx)
                anchor = mx if (anchor is None or mx > anchor) else anchor
    return anchor

def day_aligned_month_span_from_anchor_ms(anchor_ms: int, end_months_ago: int, cover_months: int, tz: str):
    """按“日对齐”的两个月区间"""
    anchor_utc = pd.to_datetime(int(anchor_ms), unit="ms", utc=True)
    anchor_local_day0 = anchor_utc.tz_convert(tz).normalize()
    end_day_local = anchor_local_day0 - pd.DateOffset(months=end_months_ago)
    start_day_local = end_day_local - pd.DateOffset(months=cover_months)
    start_local = start_day_local
    end_local = end_day_local + pd.DateOffset(days=1)
    start_utc = start_local.tz_convert("UTC")
    end_utc = end_local.tz_convert("UTC")
    return int(start_utc.value // 10**6), int(end_utc.value // 10**6)

# ===== 清洗 / 分段 / 规范化 / 滑窗 =====
def clean_and_select_columns(df):
    """清洗数据并选择必要列"""
    miss = [c for c in COL_MAP if c not in df.columns]
    if miss:
        raise KeyError(f"缺失必须字段: {miss}")
    x = df[list(COL_MAP.keys())].rename(columns=COL_MAP)
    
    # 时间列处理
    t_ms = pd.to_numeric(x['time'], errors='coerce').astype('Int64')
    x['time_ms'] = t_ms
    x['time'] = (t_ms // 1000).astype('Int64')
    
    # 数值列处理
    for c in ['mileage','volt','current','soc','max_single_volt','min_single_volt','max_temp','min_temp','cells']:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    
    # 基础清洗
    x['status'] = x['status'].astype(str).str.strip()
    x.loc[~x['soc'].between(0,100), 'soc'] = np.nan
    x.loc[x['volt'] <= 0, 'volt'] = np.nan
    x.loc[x['max_single_volt'] <= 0, 'max_single_volt'] = np.nan
    x.loc[x['min_single_volt'] <= 0, 'min_single_volt'] = np.nan
    x.loc[x['max_temp'] < -100, 'max_temp'] = np.nan
    x.loc[x['min_temp'] < -100, 'min_temp'] = np.nan
    
    # 平均单体电压（避免除零）
    x['cells'] = x['cells'].where(x['cells'] > 0, np.nan)
    x['volt'] = x['volt'] / x['cells'].replace(0, np.nan)  # 防止除零
    
    # 关键列不可为 NaN
    x = x.dropna(subset=['time','time_ms','volt','current','soc','max_single_volt','min_single_volt','max_temp','min_temp'])
    
    # 仅保留必要列
    x = x[['time','time_ms','mileage','status','volt','current','soc','max_single_volt','min_single_volt','max_temp','min_temp']].copy()
    x['time'] = x['time'].astype(np.int64)
    x['time_ms'] = x['time_ms'].astype(np.int64)
    return x

def split_charging_sessions(df):
    """按时间断点分割充电段"""
    if df.empty:
        return []
    df = df.sort_values('time').drop_duplicates(subset='time', keep='first').reset_index(drop=True)
    sid = (df['time'].diff().fillna(SAMPLE_STEP_SEC) > GAP_BREAK_SEC).cumsum()
    out = []
    for _, g in df.groupby(sid, sort=True):
        if len(g) < MIN_SEG_POINTS:
            continue
        if USE_SOC_LIMIT:
            if not (g['soc'].iloc[0] < SOC_START_LT and g['soc'].iloc[-1] > SOC_END_GT):
                continue
        out.append(g.reset_index(drop=True))
    return out

def normalize_segment_to_30s_light(seg: pd.DataFrame):
    """段内30s轻量规范化"""
    seg = seg.sort_values('time').drop_duplicates('time').reset_index(drop=True)
    if len(seg) < 2:
        return []
    out = [seg.iloc[0].copy()]
    chunks = []
    filled = 0
    for i in range(1, len(seg)):
        prev = out[-1]
        cur = seg.iloc[i]
        dt = int(cur['time'] - prev['time'])
        if dt == 0:
            continue  # 重复时间戳
        if abs(dt - STEP) <= TOL:
            out.append(cur); continue
        if abs(dt - 2*STEP) <= TOL and filled < FILL_LIMIT:
            alpha = 0.5
            mid = prev.copy()
            mid['time'] = prev['time'] + STEP
            mid['time_ms'] = prev['time_ms'] + STEP * 1000
            for c in COLS_FEAT:
                mid[c] = (1 - alpha) * float(prev[c]) + alpha * float(cur[c])
            out.append(mid)
            out.append(cur)
            filled += 1
            continue
        # 大间断/非常规：切段
        if len(out) >= 2:
            chunks.append(pd.DataFrame(out, columns=seg.columns))
        out = [cur.copy()]
        filled = 0
    if len(out) >= 2:
        chunks.append(pd.DataFrame(out, columns=seg.columns))
    return chunks

def sliding_window_stream(seg: pd.DataFrame, window_len: int, interval: int,
                          label: str, car: str, charge_number: int, mileage_start: float):
    """流式滑窗生成器"""
    seg = seg[REQ_COLS].reset_index(drop=True)
    if len(seg) < window_len:
        return
    max_start = (len(seg) - window_len) // interval
    feat = COLS_FEAT
    for i in range(max_start + 1):
        s, e = i*interval, i*interval + window_len
        if e > len(seg):
            break
        win_feat = seg.iloc[s:e][feat].reset_index(drop=True)
        if win_feat.isna().any().any():
            continue
        win_time = seg.iloc[s:e]['time_ms'].to_numpy()
        # 区间字符串
        soc0 = float(win_feat['soc'].iloc[0]); soc1 = float(win_feat['soc'].iloc[-1])
        soc_range = f"{int(round(soc0))}-{int(round(soc1))}"
        v0 = float(win_feat['volt'].iloc[0]); v1 = float(win_feat['volt'].iloc[-1])
        volt_range = f"{v0:.1f}-{v1:.1f}"
        meta = OrderedDict(
            label=label,
            car=car,
            charge_number=charge_number,
            mileage=round(float(mileage_start), 1) if pd.notna(mileage_start) else 0.0,
            ts_start_ms=int(win_time[0]),
            soc_range=soc_range,
            volt_range=volt_range
        )
        win_np = win_feat.to_numpy(dtype=np.float32)
        seq_id = np.arange(1, window_len+1, dtype=np.float32).reshape(-1,1)
        yield (np.hstack([win_np, seq_id]), meta)

# ===== 单文件处理 =====
def preprocess_one_file_stream(car, pq_path, range_ms, window_len, interval, label):
    """处理单个parquet文件，生成窗口数据"""
    try:
        raw = pd.read_parquet(pq_path, engine='pyarrow')
    except Exception:
        raw = pd.read_parquet(pq_path)  # 自动适配引擎
    df = clean_and_select_columns(raw)
    del raw  # 释放内存
    
    # 按锚点区间裁剪
    start_ms, end_ms = range_ms
    df = df[(df['time_ms'] >= start_ms) & (df['time_ms'] < end_ms)]
    if df.empty:
        return
    
    # 里程阈值过滤
    df = df[df['mileage'] >= 1000.0]
    if df.empty:
        return
    
    # 充电状态过滤
    cand = df[(df['status'] == STATUS_VALUE) & (df['current'] > CURR_THRESH)].copy()
    if cand.empty:
        return
    
    # 分段处理
    seg_idx = 0
    for big in split_charging_sessions(cand):
        seg_idx += 1
        mileage0 = float(big.iloc[0]['mileage']) if 'mileage' in big.columns else 0.0
        use = big[['time','time_ms'] + COLS_FEAT].copy()
        sub_segments = normalize_segment_to_30s_light(use)
        # 子段滑窗
        for sseg in sub_segments:
            if len(sseg) < window_len:
                continue
            for win in sliding_window_stream(
                sseg, window_len, interval,
                label=label, car=car, charge_number=seg_idx, mileage_start=mileage0
            ):
                yield win

# ===== 并行处理 =====
def worker(pair, out_folder, car_ranges, window_len, interval, label):
    """工作进程：处理文件并保存结果"""
    car, pq = pair
    saved = 0
    try:
        # 处理 DBFS 输出路径
        if out_folder.startswith("dbfs:/"):
            local_out = "/dbfs" + out_folder[5:]
        else:
            local_out = out_folder
        os.makedirs(local_out, exist_ok=True)
        
        idx = 0
        for (win_np, meta) in preprocess_one_file_stream(car, pq, car_ranges[car], window_len, interval, label):
            out_name = f'{car}_{meta["charge_number"]}_{idx}.pkl'
            out_path = os.path.join(local_out, out_name)
            torch.save((win_np, meta), out_path)
            saved += 1
            idx += 1
        return (car, saved)
    except Exception as e:
        print(f"[FAIL] {car} | {pq} | {e}")
        return (car, 0)

# ===== 主函数 =====
def main():
    # 从 widgets 读取参数，并手动转换为整数类型
    args = {
        "raw_folder": dbutils.widgets.get("raw_folder"),
        "out_folder": dbutils.widgets.get("out_folder"),
        "window_len": int(dbutils.widgets.get("window_len")),  # 手动转换为int
        "interval": int(dbutils.widgets.get("interval")),     # 手动转换为int
        "label": dbutils.widgets.get("label"),
        "end_months_ago": int(dbutils.widgets.get("end_months_ago")),  # 手动转换为int
        "cover_months": int(dbutils.widgets.get("cover_months")),      # 手动转换为int
        "tz": dbutils.widgets.get("tz"),
        "cst_tz": dbutils.widgets.get("cst_tz"),
        "car_whitelist": dbutils.widgets.get("car_whitelist"),
        "max_files": int(dbutils.widgets.get("max_files")),    # 手动转换为int
        "jobs": int(dbutils.widgets.get("jobs"))               # 手动转换为int
    }
    
    # 初始化输出目录
    if args["out_folder"].startswith("dbfs:/"):
        local_out = "/dbfs" + args["out_folder"][5:]
    else:
        local_out = args["out_folder"]
    os.makedirs(local_out, exist_ok=True)
    
    # 1) 列车、算锚点与区间
    car_dirs = list_car_dirs(args["raw_folder"], car_whitelist=args["car_whitelist"] or None)
    total_cars = len(car_dirs)
    car_ranges, pairs, cars_with_anchor = {}, [], []
    report_rows = []
    
    for car_dir in car_dirs:
        car = os.path.basename(car_dir)
        anchor = detect_car_anchor_ms(car_dir)
        if anchor is None:
            continue
        cars_with_anchor.append(car)
        start_ms, end_ms = day_aligned_month_span_from_anchor_ms(
            anchor, args["end_months_ago"], args["cover_months"], args["tz"]
        )
        car_ranges[car] = (start_ms, end_ms)
        report_rows.append({
            "car": car,
            "anchor_ms": int(anchor),
            "anchor_UTC": pretty(anchor, "UTC"),
            "anchor_CST": pretty(anchor, args["cst_tz"]),
            "start_ms": int(start_ms),
            "end_ms": int(end_ms),
            "range_UTC": f"{pretty(start_ms,'UTC')} ~ {pretty(end_ms,'UTC')}",
            "range_CST": f"{pretty(start_ms,args['cst_tz'])} ~ {pretty(end_ms,args['cst_tz'])}",
        })
        # 收集文件对
        for pq in list_parquet_files_of_car(car_dir):
            pairs.append((car, pq))
    
    # 2) 显示汇总报表（Databricks 中用 display()）
    if report_rows:
        df_report = pd.DataFrame(report_rows)
        print("车辆时间范围汇总：")
        display(df_report)  # Databricks 显示表格
    else:
        print("未检测到有效车辆或锚点")
    
    # 3) 并行处理
    if args["max_files"] > 0:
        pairs = pairs[:args["max_files"]]
    jobs_input = int(dbutils.widgets.get("jobs"))
    num_workers = max(1, min(jobs_input, os.cpu_count() or 4))
    print(f"开始处理：{len(pairs)} 个文件，使用 {num_workers} 个进程")
    
    run = partial(
        worker,
        out_folder=args["out_folder"],
        car_ranges=car_ranges,
        window_len=args["window_len"],
        interval=args["interval"],
        label=args["label"]
    )
    
    saved_counts_by_car = defaultdict(int)
    total_saved_windows = 0
    
    # 多进程处理（限制子进程任务数，避免内存泄漏）
    with Pool(processes=num_workers, maxtasksperchild=4) as pool:
        pbar = tqdm(total=len(pairs), desc="Processing")
        for car, saved in pool.imap_unordered(run, pairs, chunksize=2):
            if car is not None:
                saved_counts_by_car[car] += int(saved)
                total_saved_windows += int(saved)
            pbar.update(1)
        pbar.close()
    
    # 4) 统计结果
    cars_with_anchor_set = set(cars_with_anchor)
    cars_generated = {c for c, n in saved_counts_by_car.items() if n > 0}
    cars_zero = cars_with_anchor_set - cars_generated
    
    print("\n========== 统计结果 ==========")
    print(f"总车辆数：{total_cars}")
    print(f"检测到锚点的车辆数：{len(cars_with_anchor_set)}")
    print(f"产生窗口的车辆数：{len(cars_generated)}")
    print(f"无窗口的车辆数：{len(cars_zero)}")
    print(f"总窗口数：{total_saved_windows}")
    
    print("\n逐车窗口数量：")
    stats = pd.DataFrame([
        {"car": car, "windows": saved_counts_by_car.get(car, 0)}
        for car in sorted(cars_with_anchor_set)
    ])
    display(stats)  # Databricks 显示统计表格
    
    print(f"\n完成：结果保存至 {args['out_folder']}")

if __name__ == "__main__":
    main()