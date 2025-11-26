# Databricks notebook source
# -*- coding: utf-8 -*-
"""
Databricks 单机多进程优化版本
针对31GB/988车/24核环境深度优化：
1. 使用multiprocessing.Pool充分利用24核
2. 避免Spark开销，直接pandas处理
3. 优化I/O：批量读取、内存映射
4. 实时进度显示（适配Databricks）
5. 内存管理优化
"""

import os
import glob
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
import time
from datetime import datetime

# ===== 固定口径 =====
SAMPLE_STEP_SEC = 30
GAP_BREAK_SEC = 300
CURR_THRESH = 0.01
MIN_SEG_POINTS = 25
STATUS_VALUE = "CHARGING_IN_PARKING"
USE_SOC_LIMIT = False
SOC_START_LT = 50.0
SOC_END_GT = 65.0

# ===== 30s 轻量规范化参数 =====
STEP = 30
TOL = 3
FILL_LIMIT = 1
HARD_GAP = 90

COLS_FEAT = ['volt', 'current', 'soc',
             'max_single_volt', 'min_single_volt',
             'max_temp', 'min_temp']
REQ_COLS = ['time', 'time_ms'] + COLS_FEAT

# ===== 配置参数 =====
INPUT_PATH = '/Volumes/conf/dl/vol_prediction-rdeb-yifshen_common_blob/203_Validation_new/'
OUTPUT_PATH = '/Volumes/conf/dl/vol_prediction-rdeb-yifshen_common_blob/203_Validation_new_pkl/'
WINDOW_LEN = 40
INTERVAL = 2
LABEL = "0"
END_MONTHS_AGO = 0
COVER_MONTHS = 36
TZ = "UTC"

# 并行配置
NUM_WORKERS = 22  # 24核留2个给系统，用22个
CHUNK_SIZE = 1  # 每个任务处理1辆车


# ===== 列映射 =====
COL_MAP = {
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
}


def clean_and_select_columns(df):
    """清洗和选择列"""
    # 检查必需列
    miss = [c for c in COL_MAP if c not in df.columns]
    if miss:
        raise KeyError(f"缺失必须字段: {miss}")
    
    x = df[list(COL_MAP.keys())].rename(columns=COL_MAP)
    
    # 时间列
    t_ms = pd.to_numeric(x['time'], errors='coerce').astype('Int64')
    x['time_ms'] = t_ms
    x['time'] = (t_ms // 1000).astype('Int64')
    
    # 数值列
    for c in ['mileage', 'volt', 'current', 'soc',
              'max_single_volt', 'min_single_volt',
              'max_temp', 'min_temp', 'cells']:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    
    # 基础清洗
    x['status'] = x['status'].astype(str).str.strip()
    x.loc[~x['soc'].between(0, 100), 'soc'] = np.nan
    x.loc[x['volt'] <= 0, 'volt'] = np.nan
    x.loc[x['max_single_volt'] <= 0, 'max_single_volt'] = np.nan
    x.loc[x['min_single_volt'] <= 0, 'min_single_volt'] = np.nan
    x.loc[x['max_temp'] < -100, 'max_temp'] = np.nan
    x.loc[x['min_temp'] < -100, 'min_temp'] = np.nan
    
    # 平均单体电压
    x['cells'] = x['cells'].where(x['cells'] > 0, np.nan)
    x['volt'] = x['volt'] / x['cells']
    
    # 关键列不可为 NaN
    x = x.dropna(subset=['time', 'time_ms', 'volt', 'current', 'soc',
                         'max_single_volt', 'min_single_volt',
                         'max_temp', 'min_temp'])
    
    # 仅保留必要列
    x = x[['time', 'time_ms', 'mileage', 'status',
           'volt', 'current', 'soc',
           'max_single_volt', 'min_single_volt',
           'max_temp', 'min_temp']].copy()
    
    x['time'] = x['time'].astype(np.int64)
    x['time_ms'] = x['time_ms'].astype(np.int64)
    
    return x


def normalize_segment_to_30s_light(seg_df):
    """段内30s规范化"""
    if len(seg_df) < 2:
        return []
    
    seg_df = seg_df.sort_values('time').drop_duplicates('time').reset_index(drop=True)
    out = [seg_df.iloc[0].copy()]
    chunks = []
    filled = 0
    
    for i in range(1, len(seg_df)):
        prev = out[-1]
        cur = seg_df.iloc[i]
        dt = int(cur['time'] - prev['time'])
        
        if dt == 0:
            continue
        
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
        
        if len(out) >= 2:
            chunks.append(pd.DataFrame(out, columns=seg_df.columns))
        out = [cur.copy()]
        filled = 0
    
    if len(out) >= 2:
        chunks.append(pd.DataFrame(out, columns=seg_df.columns))
    
    return chunks


def sliding_window_stream(seg, window_len, interval, label, car, charge_number, mileage_start):
    """滑动窗口生成"""
    seg = seg[REQ_COLS].reset_index(drop=True)
    if len(seg) < window_len:
        return []
    
    feat = COLS_FEAT
    max_start = (len(seg) - window_len) // interval
    results = []
    
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
        seq_id = np.arange(1, window_len+1, dtype=np.float32).reshape(-1, 1)
        win_with_seq = np.hstack([win_np, seq_id])
        
        results.append((win_with_seq, meta, i))
    
    return results


def detect_car_anchor_ms(car_folder):
    """检测车辆锚点时间"""
    anchor = None
    parquet_files = glob.glob(os.path.join(car_folder, '*.parquet'))
    
    for pq in parquet_files:
        try:
            s = pd.read_parquet(pq, columns=['time'], engine='pyarrow')['time']
            s = pd.to_numeric(s, errors='coerce')
            if s.size:
                mx = s.max(skipna=True)
                if pd.notna(mx):
                    mx = int(mx)
                    anchor = mx if (anchor is None or mx > anchor) else anchor
        except Exception:
            continue
    
    return anchor


def day_aligned_month_span_from_anchor_ms(anchor_ms, end_months_ago, cover_months, tz):
    """计算时间范围"""
    anchor_utc = pd.to_datetime(int(anchor_ms), unit="ms", utc=True)
    anchor_local_day0 = anchor_utc.tz_convert(tz).normalize()
    end_day_local = anchor_local_day0 - pd.DateOffset(months=end_months_ago)
    start_day_local = end_day_local - pd.DateOffset(months=cover_months)
    start_local = start_day_local
    end_local = end_day_local + pd.DateOffset(days=1)
    start_utc = start_local.tz_convert("UTC")
    end_utc = end_local.tz_convert("UTC")
    return int(start_utc.value // 10**6), int(end_utc.value // 10**6)


def process_one_car(car_info, output_path, progress_dict, lock):
    """
    处理单个车辆的所有数据
    car_info: (car_name, car_folder, start_ms, end_ms)
    """
    car_name, car_folder, start_ms, end_ms = car_info
    
    try:
        # 读取该车辆的所有parquet文件
        parquet_files = glob.glob(os.path.join(car_folder, '*.parquet'))
        if not parquet_files:
            return (car_name, 0, 0)
        
        # 合并读取所有parquet文件
        dfs = []
        for pq in parquet_files:
            try:
                df = pd.read_parquet(pq, engine='pyarrow')
                dfs.append(df)
            except Exception as e:
                continue
        
        if not dfs:
            return (car_name, 0, 0)
        
        # 合并数据
        raw = pd.concat(dfs, ignore_index=True)
        del dfs
        
        # 清洗数据
        df = clean_and_select_columns(raw)
        del raw
        
        # 时间范围过滤
        df = df[(df['time_ms'] >= start_ms) & (df['time_ms'] < end_ms)]
        if df.empty:
            return (car_name, 0, 0)
        
        # 里程过滤
        df = df[df['mileage'] >= 1000.0]
        if df.empty:
            return (car_name, 0, 0)
        
        # 充电状态过滤
        df = df[(df['status'] == STATUS_VALUE) & (df['current'] > CURR_THRESH)]
        if df.empty:
            return (car_name, 0, 0)
        
        # 排序和去重
        df = df.sort_values('time').drop_duplicates(subset='time', keep='first').reset_index(drop=True)
        
        # 分段
        df['time_diff'] = df['time'].diff().fillna(SAMPLE_STEP_SEC)
        df['segment_id'] = (df['time_diff'] > GAP_BREAK_SEC).cumsum()
        
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 处理每个分段
        total_windows = 0
        success_count = 0
        
        for seg_id, seg_group in df.groupby('segment_id'):
            if len(seg_group) < MIN_SEG_POINTS:
                continue
            
            if USE_SOC_LIMIT:
                if not (seg_group['soc'].iloc[0] < SOC_START_LT and 
                       seg_group['soc'].iloc[-1] > SOC_END_GT):
                    continue
            
            seg_group = seg_group.reset_index(drop=True)
            mileage0 = float(seg_group.iloc[0]['mileage'])
            
            # 30s规范化
            use_cols = ['time', 'time_ms'] + COLS_FEAT
            use_df = seg_group[use_cols].copy()
            sub_segments = normalize_segment_to_30s_light(use_df)
            
            # 滑动窗口
            for sseg in sub_segments:
                if len(sseg) < WINDOW_LEN:
                    continue
                
                windows = sliding_window_stream(
                    sseg, WINDOW_LEN, INTERVAL,
                    label=LABEL, car=car_name,
                    charge_number=int(seg_id), mileage_start=mileage0
                )
                
                # 保存窗口
                for win_np, meta, win_id in windows:
                    try:
                        out_name = f"{car_name}_{meta['charge_number']}_{win_id}.pkl"
                        out_path = os.path.join(output_path, out_name)
                        torch.save((win_np, meta), out_path)
                        success_count += 1
                        total_windows += 1
                    except Exception:
                        total_windows += 1
        
        # 更新进度
        with lock:
            progress_dict['completed'] += 1
            progress_dict['total_windows'] += success_count
            current = progress_dict['completed']
            total = progress_dict['total_cars']
            windows = progress_dict['total_windows']
            
            # 每10辆车打印一次进度
            if current % 10 == 0 or current == total:
                elapsed = time.time() - progress_dict['start_time']
                rate = current / elapsed if elapsed > 0 else 0
                eta = (total - current) / rate if rate > 0 else 0
                print(f"进度: {current}/{total} 车辆 ({100*current/total:.1f}%) | "
                      f"已生成 {windows:,} 个窗口 | "
                      f"速度: {rate:.1f} 车/秒 | "
                      f"预计剩余: {eta/60:.1f} 分钟")
        
        return (car_name, success_count, 0)
    
    except Exception as e:
        print(f"处理车辆 {car_name} 失败: {e}")
        with lock:
            progress_dict['completed'] += 1
        return (car_name, 0, 1)


def main():
    """主函数"""
    print("=" * 80)
    print("Databricks 单机多进程优化版本")
    print("=" * 80)
    print(f"输入路径: {INPUT_PATH}")
    print(f"输出路径: {OUTPUT_PATH}")
    print(f"窗口长度: {WINDOW_LEN}, 间隔: {INTERVAL}")
    print(f"并行进程数: {NUM_WORKERS}")
    print(f"CPU核心数: {cpu_count()}")
    print("=" * 80)
    
    start_time = time.time()
    
    # 1. 扫描车辆文件夹
    print("\n步骤1: 扫描车辆文件夹...")
    car_folders = sorted([d for d in glob.glob(os.path.join(INPUT_PATH, '*')) 
                         if os.path.isdir(d)])
    print(f"发现 {len(car_folders)} 个车辆文件夹")
    
    # 2. 检测锚点和计算时间范围
    print("\n步骤2: 检测锚点和计算时间范围...")
    car_tasks = []
    
    for idx, car_folder in enumerate(car_folders, 1):
        car_name = os.path.basename(car_folder)
        
        try:
            anchor_ms = detect_car_anchor_ms(car_folder)
            if anchor_ms is None:
                continue
            
            start_ms, end_ms = day_aligned_month_span_from_anchor_ms(
                anchor_ms, END_MONTHS_AGO, COVER_MONTHS, TZ
            )
            
            car_tasks.append((car_name, car_folder, start_ms, end_ms))
            
            if idx % 50 == 0:
                print(f"  已处理 {idx}/{len(car_folders)} 个车辆...")
        
        except Exception as e:
            continue
    
    print(f"✓ 共有 {len(car_tasks)} 个有效车辆")
    
    if not car_tasks:
        print("没有有效数据，退出")
        return
    
    # 3. 多进程并行处理
    print("\n步骤3: 开始多进程并行处理...")
    print("=" * 80)
    
    # 创建共享进度字典
    manager = Manager()
    progress_dict = manager.dict()
    progress_dict['completed'] = 0
    progress_dict['total_cars'] = len(car_tasks)
    progress_dict['total_windows'] = 0
    progress_dict['start_time'] = time.time()
    lock = manager.Lock()
    
    # 创建进程池
    with Pool(processes=NUM_WORKERS) as pool:
        # 使用partial固定参数
        worker_func = partial(process_one_car, 
                             output_path=OUTPUT_PATH,
                             progress_dict=progress_dict,
                             lock=lock)
        
        # 并行处理
        results = pool.map(worker_func, car_tasks, chunksize=CHUNK_SIZE)
    
    # 4. 统计结果
    print("\n\n步骤4: 统计结果...")
    
    total_windows = sum(r[1] for r in results)
    total_failed = sum(r[2] for r in results)
    cars_with_windows = sum(1 for r in results if r[1] > 0)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("处理完成！统计结果：")
    print("=" * 80)
    print(f"总车辆数（扫描到）: {len(car_folders)}")
    print(f"有效车辆数（有锚点）: {len(car_tasks)}")
    print(f"生成窗口的车辆数: {cars_with_windows}")
    print(f"总窗口数: {total_windows:,}")
    print(f"失败车辆数: {total_failed}")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"平均速度: {len(car_tasks)/elapsed:.2f} 车/秒")
    print(f"窗口生成速度: {total_windows/elapsed:.0f} 窗口/秒")
    
    print("\n各车辆窗口数量（前20辆）：")
    print("-" * 80)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    for i, (car_name, window_count, failed) in enumerate(sorted_results[:20], 1):
        if window_count > 0:
            print(f"{i}. {car_name}: {window_count:,} 个窗口")
    
    print("\n" + "=" * 80)
    print(f"所有文件已保存到: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()