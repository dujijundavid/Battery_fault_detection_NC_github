import os, glob, argparse, re
from collections import OrderedDict, defaultdict
from multiprocessing import Pool
from functools import partial
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
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
COLS_FEAT = ['volt','current','soc',
             'max_single_volt','min_single_volt',
             'max_temp','min_temp']
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
# ===== 小工具 =====
def pretty(ts_ms, tz="UTC"):
    """将UTC毫秒时间戳格式化为可读字符串"""
    return pd.to_datetime(int(ts_ms), unit="ms", utc=True) \
             .tz_convert(tz) \
             .strftime("%Y-%m-%d %H:%M:%S %Z")
def list_car_dirs(root, car_whitelist=None):
    subs = [d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d)]
    if car_whitelist:
        if any(ch in car_whitelist for ch in '*.?|[]()+\\^$'):
            # 正则
            pat = re.compile(car_whitelist)
            subs = [d for d in subs if pat.search(os.path.basename(d))]
        else:
            # 逗号白名单
            wl = {s.strip() for s in car_whitelist.split(',') if s.strip()}
            subs = [d for d in subs if os.path.basename(d) in wl]
    return sorted(subs)
def list_parquet_files_of_car(car_dir):
    return sorted(glob.glob(os.path.join(car_dir, '*.parquet')))
def detect_car_anchor_ms(car_dir):
    """读取该车所有parquet的time列最大值(UTC毫秒)为锚点；失败返回None。"""
    anchor = None
    for pq in list_parquet_files_of_car(car_dir):
        try:
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
def day_aligned_month_span_from_anchor_ms(anchor_ms: int,
                                          end_months_ago: int,
                                          cover_months: int,
                                          tz: str):
    """
    按“日对齐”的两个月区间（见文件头注释）
    """
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
def clean_and_select_columns(df: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in COL_MAP if c not in df.columns]
    if miss:
        raise KeyError(f"缺失必须字段: {miss}")
    x = df[list(COL_MAP.keys())].rename(columns=COL_MAP)
    # 时间列
    t_ms = pd.to_numeric(x['time'], errors='coerce').astype('Int64')
    x['time_ms'] = t_ms
    x['time'] = (t_ms // 1000).astype('Int64')
    # 数值列
    for c in ['mileage','volt','current','soc',
              'max_single_volt','min_single_volt',
              'max_temp','min_temp','cells']:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    # 基础清洗
    x['status'] = x['status'].astype(str).str.strip()
    # x.loc[~x['soc'].between(0,100), 'soc'] = np.nan
    # x.loc[x['volt'] <= 0, 'volt'] = np.nan
    # x.loc[x['max_single_volt'] <= 0, 'max_single_volt'] = np.nan
    # x.loc[x['min_single_volt'] <= 0, 'min_single_volt'] = np.nan
    # x.loc[x['max_temp'] < -100, 'max_temp'] = np.nan
    # x.loc[x['min_temp'] < -100, 'min_temp'] = np.nan
    mask_soc = ~x['soc'].between(0, 100)
    mask_volt = x['volt'] <= 0
    mask_max_single_volt = x['max_single_volt'] <= 0
    mask_min_single_volt = x['min_single_volt'] <= 0
    mask_max_temp = x['max_temp'] < -100
    mask_min_temp = x['min_temp'] < -100

    x.loc[mask_soc, 'soc'] = np.nan
    x.loc[mask_volt, 'volt'] = np.nan
    x.loc[mask_max_single_volt, 'max_single_volt'] = np.nan
    x.loc[mask_min_single_volt, 'min_single_volt'] = np.nan
    x.loc[mask_max_temp, 'max_temp'] = np.nan
    x.loc[mask_min_temp, 'min_temp'] = np.nan

    # 额外优化：dropna时指定axis=0，明确行删除
    x = x.dropna(subset=['time','time_ms','volt','current','soc',
                        'max_single_volt','min_single_volt',
                        'max_temp','min_temp'], axis=0)
    # 平均单体电压
    x['cells'] = x['cells'].where(x['cells'] > 0, np.nan)
    x['volt'] = x['volt'] / x['cells']
    # 关键列不可为 NaN
    x = x.dropna(subset=['time','time_ms','volt','current','soc',
                         'max_single_volt','min_single_volt',
                         'max_temp','min_temp'])
    # 仅保留必要列
    x = x[['time','time_ms','mileage','status',
           'volt','current','soc',
           'max_single_volt','min_single_volt',
           'max_temp','min_temp']].copy()
    x['time'] = x['time'].astype(np.int64)
    x['time_ms'] = x['time_ms'].astype(np.int64)
    return x
def split_charging_sessions(df: pd.DataFrame):
    if df.empty:
        return []
    df = df.sort_values('time') \
           .drop_duplicates(subset='time', keep='first') \
           .reset_index(drop=True)
    sid = (df['time'].diff().fillna(SAMPLE_STEP_SEC) > GAP_BREAK_SEC).cumsum()
    out = []
    for _, g in df.groupby(sid, sort=True):
        if len(g) < MIN_SEG_POINTS:
            continue
        if USE_SOC_LIMIT:
            if not (g['soc'].iloc[0] < SOC_START_LT and
                    g['soc'].iloc[-1] > SOC_END_GT):
                continue
        out.append(g.reset_index(drop=True))
    return out
def normalize_segment_to_30s_light(seg: pd.DataFrame):
    """
    段内“轻量规范化”：见文件头，返回 list[pd.DataFrame]
    """
    seg = seg.sort_values('time') \
             .drop_duplicates('time') \
             .reset_index(drop=True)
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
            continue # 重复时间戳
        if abs(dt - STEP) <= TOL:
            out.append(cur)
            continue
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
def sliding_window_stream(seg: pd.DataFrame,
                          window_len: int,
                          interval: int,
                          label: str,
                          car: str,
                          charge_number: int,
                          mileage_start: float):
    """
    直接按行滑窗，逐个 yield (win_np, meta)；只保留满窗，不做段首补齐。
    同时在 meta 中写入 soc_range、volt_range。
    """
    seg = seg[REQ_COLS].reset_index(drop=True)
    if len(seg) < window_len:
        return
    feat = COLS_FEAT
    max_start = (len(seg) - window_len) // interval
    for i in range(max_start + 1):
        s, e = i*interval, i*interval + window_len
        if e > len(seg):
            break
        win_feat = seg.iloc[s:e][feat].reset_index(drop=True)
        if win_feat.isna().any().any():
            continue
        win_time = seg.iloc[s:e]['time_ms'].to_numpy()
        soc0 = float(win_feat['soc'].iloc[0])
        soc1 = float(win_feat['soc'].iloc[-1])
        soc_range = f"{int(round(soc0))}-{int(round(soc1))}"
        v0 = float(win_feat['volt'].iloc[0])
        v1 = float(win_feat['volt'].iloc[-1])
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
# ===== 单文件（生成器） =====
def preprocess_one_file_stream(car, pq_path, range_ms,
                               window_len, interval, label):
    """
    读取一个 parquet，按规则逐窗生成 (win_np, meta)
    """
    # try:
    #     raw = pd.read_parquet(
    #         pq_path, engine='pyarrow',
    #         columns=['time','odo','bit_charging_state','bms_total_voltage',
    #                  'bms_total_current','bms_soc','bms_volt_max_value',
    #                  'bms_volt_min_value','bms_temp_max_value',
    #                  'bms_temp_min_value','bms_tba_cells_1']
    #     )
    # except Exception:
    #     raw = pd.read_parquet(pq_path)
    
    try:
        dtype_map = {
            'time': 'int64',
            'odo': 'float32',
            'bit_charging_state': 'str',
            'bms_total_voltage': 'float32',
            'bms_total_current': 'float32',
            'bms_soc': 'float32',
            'bms_volt_max_value': 'float32',
            'bms_volt_min_value': 'float32',
            'bms_temp_max_value': 'float32',
            'bms_temp_min_value': 'float32',
            'bms_tba_cells_1': 'int32'
        }
        raw = pd.read_parquet(
            pq_path, 
            engine='pyarrow',
            columns=list(dtype_map.keys()),
            dtype=dtype_map,
            use_threads=True  # 启用pyarrow的多线程读取
        )
    except Exception:
        raw = pd.read_parquet(pq_path, engine='pyarrow', use_threads=True)

    df = clean_and_select_columns(raw)
    del raw
    start_ms, end_ms = range_ms
    df = df[(df['time_ms'] >= start_ms) & (df['time_ms'] < end_ms)]
    if df.empty:
        return
    df = df[df['mileage'] >= 1000.0]
    if df.empty:
        return
    cand = df[(df['status'] == STATUS_VALUE) &
              (df['current'] > CURR_THRESH)].copy()
    if cand.empty:
        return
    seg_idx = 0
    for big in split_charging_sessions(cand):
        seg_idx += 1
        mileage0 = float(big.iloc[0]['mileage']) if 'mileage' in big.columns else 0.0
        use = big[['time','time_ms'] + COLS_FEAT].copy()
        sub_segments = normalize_segment_to_30s_light(use)
        for sseg in sub_segments:
            if len(sseg) < window_len:
                continue
            for win in sliding_window_stream(
                sseg, window_len, interval,
                label=label, car=car,
                charge_number=seg_idx, mileage_start=mileage0
            ):
                yield win
# ===== worker（边产出边保存） =====
def worker(pair, out_folder, car_ranges, window_len, interval, label):
    car, pq = pair
    saved = 0
    try:
        idx = 0
        for (win_np, meta) in preprocess_one_file_stream(
            car, pq, car_ranges[car],
            window_len, interval, label
        ):
            out_name = f'{car}_{meta["charge_number"]}_{idx}.pkl'
            out_path = os.path.join(out_folder, out_name)
            torch.save((win_np, meta), out_path)
            saved += 1
            idx += 1
        return (car, saved)
    except Exception as e:
        print(f"[FAIL] {car} | {pq} | {e}")
        return (car, 0)
# ===== 主程序 =====
def main():
    ap = argparse.ArgumentParser(
        description="每车按日对齐的相对月份筛选 ➜ 充电段(30s规范化) ➜ 流式滑窗 ➜ .pkl + 车辆统计"
    )
    ap.add_argument("--raw_folder", type=str, required=False,
                    default=r"/Volumes/conf/dl/vol_prediction-rdeb-yifshen_common_blob/203_Validation_new/",
                    help="根目录（下含 车号/*.parquet）")
    ap.add_argument("--out_folder", type=str, required=False,
                    default=r"/Volumes/conf/dl/vol_prediction-rdeb-yifshen_common_blob/203_Validation_pkl1/",
                    help="输出目录")
    ap.add_argument("--window_len", type=int, default=40)
    ap.add_argument("--interval", type=int, default=2)
    ap.add_argument("--label", type=str, default="0")
    ap.add_argument("--end_months_ago", type=int, default=0,
                    help="结束日=锚点日期往前该值月（按日对齐）")
    ap.add_argument("--cover_months", type=int, default=12,
                    help="向前覆盖的月数（含结束日，按日对齐）")
    ap.add_argument("--tz", type=str, default="UTC",
                    help="日界线所在时区（如 UTC / Asia/Taipei / Asia/Shanghai）")
    ap.add_argument("--car_whitelist", type=str, default="",
                    help="逗号白名单或正则")
    ap.add_argument("--max_files", type=int, default=None)
    # ap.add_argument("--jobs", type=int,
    #                 default=min(12, os.cpu_count() or 8)) # 多线程运行
    ap.add_argument("--jobs", type=int,
                    default=22)  # 单进程运行

    #args = ap.parse_args()#改
    args, _ = ap.parse_known_args()
    # 确保输出目录存在
    os.makedirs(args.out_folder, exist_ok=True)
    # 1) 列车、算锚点与区间
    car_dirs = list_car_dirs(args.raw_folder,
                             car_whitelist=args.car_whitelist or None)
    total_cars = len(car_dirs)
    car_ranges, pairs, cars_with_anchor = {}, [], []
    for car_dir in car_dirs:
        car = os.path.basename(car_dir)
        anchor = detect_car_anchor_ms(car_dir)
        if anchor is None:
            continue
        cars_with_anchor.append(car)
        start_ms, end_ms = day_aligned_month_span_from_anchor_ms(
            anchor, args.end_months_ago, args.cover_months, args.tz
        )
        car_ranges[car] = (start_ms, end_ms)
        for pq in list_parquet_files_of_car(car_dir):
            pairs.append((car, pq))
    if args.max_files is not None:
        pairs = pairs[:args.max_files]
    num_workers = max(1, min(args.jobs, os.cpu_count() or 8))
    print(f"开始并行处理：{len(pairs)} 个文件，使用 {num_workers} 个进程")
    run = partial(worker,
                  out_folder=args.out_folder,
                  car_ranges=car_ranges,
                  window_len=args.window_len,
                  interval=args.interval,
                  label=args.label)
    saved_counts_by_car = defaultdict(int)
    total_saved_windows = 0
    with Pool(processes=num_workers, maxtasksperchild=8) as pool:
        pbar = tqdm(total=len(pairs), desc="Processing")
        for car, saved in pool.imap_unordered(run, pairs, chunksize=16):
            if car is not None:
                saved_counts_by_car[car] += int(saved)
                total_saved_windows += int(saved)
            pbar.update(1)
        pbar.close()
    cars_with_anchor_set = set(cars_with_anchor)
    cars_generated = {c for c, n in saved_counts_by_car.items() if n > 0}
    cars_zero = cars_with_anchor_set - cars_generated
    print("\n========== 车辆统计 ==========")
    print(f"总车辆数（目录下）：{total_cars}")
    print(f"检测到锚点的车辆数：{len(cars_with_anchor_set)}")
    print(f"满足条件（产生≥1窗口）的车辆数：{len(cars_generated)}")
    print(f"被排除（有锚点但产生0窗口）的车辆数：{len(cars_zero)}")
    print(f"总窗口数：{total_saved_windows}")
    print("\n—— 逐车窗口数量（car -> windows）——")
    for car in sorted(cars_with_anchor_set):
        print(f"{car} -> {saved_counts_by_car.get(car, 0)}")
    print("\n完成：生成 {} 个 .pkl 窗口文件".format(total_saved_windows))
if __name__ == "__main__":
    main()