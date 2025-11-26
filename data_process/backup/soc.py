# -*- coding: utf-8 -*-
"""
SOC segment distribution (all charging segments, no filtering) + per-segment summary.

Changes:
- segments_soc_coverage.csv:
  * 2nd column: mileage (segment start)
  * 4th/5th: start/end time in ISO UTC
  * next: duration_min
  * covered_bins: "min_soc-max_soc" (integers)
- Optional SOC filter for distribution stats (e.g., only 30–80%).
- Extra plot: mileage_distribution.png (English labels)

Note: Excel may auto-convert "9-94" to "Sep-94". Set EXCEL_SAFE_RANGE_STR=True to output "9_to_94".
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============== Paths & Config ===============
IN_ROOT  = r"C:\Users\YIFSHEN\Documents\01_InputRawData\test_a"
OUT_DIR  = r"C:\Users\YIFSHEN\Documents\01_InputRawData\test_a"
PARQUET_GLOB = "*.parquet"

# Column candidates
CANDIDATE_TIME_COLS    = ["time", "timestamp", "gmt_time", "time_stamp"]
CANDIDATE_SOC_COLS     = ["bms_soc", "soc", "SOC"]
CANDIDATE_CURRENT_COLS = ["bms_total_current", "current", "pack_current"]
CANDIDATE_CHGFLAG_COLS = ["bit_charging_state", "charging_flag", "charging_in_parking", "is_chg_parking"]
CANDIDATE_MILEAGE_COLS = ["odo", "mileage", "odometer", "total_mileage"]

# Segment rules
BREAK_GAP_SECONDS = 600     # cut when Δt > 600 s
CURRENT_MIN       = 0.01    # charging threshold (A)

# SOC bins for overall distribution
SOC_MIN, SOC_MAX = 0, 100
BIN_STEP         = 5

# Optional SOC filter for stats (put 30/80 to count only that range)
APPLY_SOC_FILTER = False
SOC_FILTER_MIN   = 30.0
SOC_FILTER_MAX   = 80.0

# Excel safety for covered_bins (avoid "Sep-94" issue)
EXCEL_SAFE_RANGE_STR = False  # True -> "30_to_80", False -> "30-80"
# =============================================

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def pick_first_existing(cands, df):
    for c in cands:
        if c in df.columns:
            return c
    return None

def _detect_is_ms(t_series: pd.Series) -> bool:
    v = t_series.to_numpy(dtype=float, copy=False)
    if v.size <= 1:
        return np.nanmax(v) > 1e10
    dt = np.diff(v)
    med = np.nanmedian(dt)
    return bool(np.isfinite(med) and (med > 50))

def _epoch_to_iso(ts: float, is_ms: bool) -> str:
    try:
        dt = pd.to_datetime(ts, unit='ms' if is_ms else 's', utc=True)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""

def read_needed_columns(parquet_path: Path):
    try:
        df0 = pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception:
        return (None,)*6

    col_time    = pick_first_existing(CANDIDATE_TIME_COLS, df0)
    col_soc     = pick_first_existing(CANDIDATE_SOC_COLS, df0)
    col_current = pick_first_existing(CANDIDATE_CURRENT_COLS, df0)
    col_flag    = pick_first_existing(CANDIDATE_CHGFLAG_COLS, df0)
    col_mileage = pick_first_existing(CANDIDATE_MILEAGE_COLS, df0)

    if (col_time is None) or (col_soc is None) or ((col_current is None) and (col_flag is None)):
        return (None,)*6

    need_cols = [c for c in [col_time, col_soc, col_current, col_flag, col_mileage] if c is not None]
    try:
        df = pd.read_parquet(parquet_path, columns=need_cols, engine="pyarrow")
    except Exception:
        return (None,)*6

    df[col_time] = pd.to_numeric(df[col_time], errors="coerce")
    df[col_soc]  = pd.to_numeric(df[col_soc],  errors="coerce")
    if col_current: df[col_current] = pd.to_numeric(df[col_current], errors="coerce")
    if col_mileage: df[col_mileage] = pd.to_numeric(df[col_mileage], errors="coerce")

    df = df.dropna(subset=[col_time, col_soc]).sort_values(col_time).reset_index(drop=True)
    if df.empty: return (None,)*6

    soc = df[col_soc].astype(float)
    if soc.max() <= 1.5: soc = soc * 100.0
    df[col_soc] = soc
    df = df[(df[col_soc] >= SOC_MIN) & (df[col_soc] <= SOC_MAX)]
    if df.empty: return (None,)*6

    return df, col_time, col_soc, col_current, col_flag, col_mileage

def detect_time_unit_and_breaks(t: pd.Series, gap_seconds: int):
    v = t.to_numpy(dtype=float, copy=False)
    if v.size == 0:
        return np.array([], dtype=bool)
    dt = np.diff(v)
    median_dt = np.nanmedian(dt) if dt.size > 0 else np.nan
    as_ms = bool(np.isfinite(median_dt) and (median_dt > 50))
    thresh = gap_seconds * (1000.0 if as_ms else 1.0)

    b = np.zeros(v.shape[0], dtype=bool)
    b[0] = True
    if dt.size > 0:
        b[1:] = dt > thresh
    return b

def iter_segments(df, col_time, col_soc, col_current=None, col_flag=None):
    m = pd.Series(True, index=df.index)

    if col_flag and (col_flag in df.columns):
        fstr = df[col_flag].astype(str).str.strip().str.upper()
        fnum = pd.to_numeric(df[col_flag], errors="coerce")
        flag_true = (fstr == "CHARGING_IN_PARKING") | (fstr == "1") | (fstr == "TRUE") | (fnum == 1)
        m &= flag_true

    if col_current and (col_current in df.columns):
        m &= (df[col_current] > CURRENT_MIN)

    dfc = df.loc[m].copy()
    if dfc.empty: return

    breaks = detect_time_unit_and_breaks(dfc[col_time], BREAK_GAP_SECONDS)
    seg_ids = breaks.cumsum()
    for sid in np.unique(seg_ids):
        seg = dfc.loc[seg_ids == sid]
        if seg.empty: continue
        yield seg

def main():
    in_root = Path(IN_ROOT)
    out_dir = Path(OUT_DIR)
    safe_mkdir(out_dir)

    # SOC bins (for overall distribution)
    edges = np.arange(SOC_MIN, SOC_MAX + BIN_STEP, BIN_STEP, dtype=float)
    if edges[-1] > SOC_MAX: edges[-1] = SOC_MAX
    nbins  = len(edges) - 1
    labels = [f"{int(l)}–{int(r)}" if (l.is_integer() and r.is_integer()) else f"{l:.1f}–{r:.1f}"
              for l, r in zip(edges[:-1], edges[1:])]

    bin_counts = np.zeros(nbins, dtype=int)
    seg_rows   = []
    mileage_starts = []

    total_segments = 0
    used_segments  = 0

    vdirs = [d for d in in_root.iterdir() if d.is_dir()]
    if not vdirs:
        print(f"[WARN] No vehicle subfolders under {in_root}")
        return

    for vdir in tqdm(vdirs, desc="Scanning vehicles"):
        vehicle_id = vdir.name
        parts = []
        col_names = None
        for pq in sorted(vdir.glob(PARQUET_GLOB)):
            pack = read_needed_columns(pq)
            if pack[0] is None: continue
            df, ctime, csoc, ccur, cflag, cmileage = pack
            if col_names is None:
                col_names = (ctime, csoc, ccur, cflag, cmileage)
            parts.append(df)

        if not parts or (col_names is None): continue

        ctime, csoc, ccur, cflag, cmileage = col_names
        car_df = pd.concat(parts, ignore_index=True)
        if car_df.empty: continue
        car_df = car_df.sort_values(ctime).reset_index(drop=True)

        seg_idx = 0
        for seg in iter_segments(car_df, ctime, csoc, ccur, cflag):
            total_segments += 1
            seg_idx += 1

            vals_soc = seg[csoc].to_numpy(dtype=float, copy=False)

            # Apply SOC filter for distribution stats (points-level)
            vals_for_stat = vals_soc
            if APPLY_SOC_FILTER:
                vals_for_stat = vals_soc[(vals_soc >= SOC_FILTER_MIN) & (vals_soc <= SOC_FILTER_MAX)]

            # contribute to histogram if any point left after filter
            if vals_for_stat.size > 0:
                used_segments += 1
                idx = np.digitize(vals_for_stat, bins=edges, right=False) - 1
                idx = idx[(idx >= 0) & (idx < nbins)]
                if idx.size > 0:
                    uniq_bins = np.unique(idx)
                    bin_counts[uniq_bins] += 1

            # per-segment summary (always keep)
            is_ms = _detect_is_ms(seg[ctime])
            t0, t1 = float(seg[ctime].iloc[0]), float(seg[ctime].iloc[-1])
            start_iso = _epoch_to_iso(t0, is_ms)
            end_iso   = _epoch_to_iso(t1, is_ms)
            dur_min   = round((t1 - t0) / (60000.0 if is_ms else 60.0), 2)

            mileage_start = (float(seg[cmileage].iloc[0]) if cmileage and cmileage in seg.columns
                             and pd.notna(seg[cmileage].iloc[0]) else np.nan)
            if np.isfinite(mileage_start):
                mileage_starts.append(mileage_start)

            min_soc = float(np.nanmin(vals_soc)) if vals_soc.size else np.nan
            max_soc = float(np.nanmax(vals_soc)) if vals_soc.size else np.nan
            lo = int(np.floor(min_soc)) if np.isfinite(min_soc) else None
            hi = int(np.ceil(max_soc))  if np.isfinite(max_soc)  else None
            if lo is not None and hi is not None:
                if EXCEL_SAFE_RANGE_STR:
                    soc_range_str = f"{lo}_to_{hi}"
                else:
                    soc_range_str = f"{lo}-{hi}"
            else:
                soc_range_str = ""

            seg_rows.append({
                "vehicle_id": vehicle_id,
                "mileage": mileage_start,
                "segment_idx": seg_idx,
                "start_time": start_iso,
                "end_time":   end_iso,
                "duration_min": dur_min,
                "len_points": int(len(seg)),
                "start_soc":  float(seg[csoc].iloc[0]),
                "end_soc":    float(seg[csoc].iloc[-1]),
                "min_soc":    min_soc,
                "max_soc":    max_soc,
                "covered_bins": soc_range_str
            })

    # ---------- Output 1: SOC overall distribution ----------
    result = pd.DataFrame({
        "soc_bin_left":  edges[:-1],
        "soc_bin_right": edges[1:],
        "soc_bin_label": labels,
        "segment_count": bin_counts
    })
    out_dir = Path(OUT_DIR); safe_mkdir(out_dir)
    csv_path = out_dir / ("soc_segment_distribution" + ("_filtered" if APPLY_SOC_FILTER else "") + ".csv")
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(14, 6), dpi=150)
    plt.bar(result["soc_bin_label"], result["segment_count"])
    plt.xlabel("SOC bin (%)")
    plt.ylabel("Segment count")
    title_extra = f" | SOC filter [{SOC_FILTER_MIN}-{SOC_FILTER_MAX}]%" if APPLY_SOC_FILTER else ""
    plt.title(f"SOC distribution by charging segments (no segment filtering){title_extra}\n"
              f"Total segments: {total_segments} | Segments contributing: {used_segments}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig_path = out_dir / ("soc_segment_distribution" + ("_filtered" if APPLY_SOC_FILTER else "") + ".png")
    plt.savefig(fig_path); plt.close()

    # ---------- Output 2: per-segment summary ----------
    seg_df = pd.DataFrame(seg_rows, columns=[
        "vehicle_id","mileage","segment_idx",
        "start_time","end_time","duration_min",
        "len_points","start_soc","end_soc","min_soc","max_soc","covered_bins"
    ])
    seg_csv = out_dir / "segments_soc_coverage.csv"
    seg_df.to_csv(seg_csv, index=False, encoding="utf-8-sig")

    # ---------- Output 3: mileage distribution (English) ----------
    if len(mileage_starts) > 0:
        plt.figure(figsize=(12, 6), dpi=150)
        plt.hist(mileage_starts, bins=50)
        plt.xlabel("Mileage at segment start")
        plt.ylabel("Segment count")
        plt.title("Mileage distribution of charging segments (segment-start mileage)")
        plt.tight_layout()
        mile_fig = out_dir / "mileage_distribution.png"
        plt.savefig(mile_fig); plt.close()
    else:
        mile_fig = "(no mileage data)"

    print("====== DONE ======")
    print(f"SOC distribution CSV: {csv_path}")
    print(f"SOC distribution PNG: {fig_path}")
    print(f"Per-segment summary : {seg_csv}")
    print(f"Mileage histogram   : {mile_fig}")

if __name__ == "__main__":
    main()
