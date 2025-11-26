# -*- coding: utf-8 -*-
import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import fnmatch

# ===== 配置 =====
IN_ROOT  = r"C:\Users\YIFSHEN\Documents\01_InputRawData\3000_70mv"
OUT_ROOT = r"C:\Users\YIFSHEN\Documents\01_InputRawData\3000_70mv_new"
START_IDX = 35          # 从 1 开始：0_0001
PREFIX = "1_"
DIGITS = 4
OVERWRITE = False      # 仅作用于 parquet 重名时是否覆盖
# =================

def is_parquet(p: Path) -> bool:
    name = p.name.lower()
    # 兼容 .snappy.parquet 这类复合后缀
    return p.is_file() and (name.endswith(".parquet") or name.endswith(".snappy.parquet"))

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_marker_files(src_dir: Path, dst_dir: Path):
    """
    将 src_dir 下的 Hadoop/Spark 标记文件复制到 dst_dir：
    - _SUCCESS
    - _started_*
    - _committed_*
    已存在则跳过；不重命名。
    """
    patterns = ["_SUCCESS", "_started_*", "_committed_*"]
    for entry in src_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            target = dst_dir / name
            if not target.exists():
                shutil.copy2(entry, target)

def main():
    in_root, out_root = Path(IN_ROOT), Path(OUT_ROOT)
    if not in_root.exists():
        print(f"[ERROR] 输入目录不存在：{in_root}")
        sys.exit(1)
    safe_mkdir(out_root)

    # 收集 parquet 并稳定排序（按全路径）
    files = sorted((p for p in in_root.rglob("*") if is_parquet(p)),
                   key=lambda x: str(x).lower())
    if not files:
        print("[INFO] 未找到 PARQUET 文件。")
        return

    rows, idx = [], START_IDX

    for src in files:
        # —— 为该 parquet 生成“同名子文件夹 + 新文件名” —— #
        vehicle_id = f"{PREFIX}{idx:0{DIGITS}d}"   # 0_0001
        subfolder = out_root / vehicle_id          # 子文件夹名：0_0001（无后缀）
        safe_mkdir(subfolder)

        new_name = f"{vehicle_id}.parquet"         # 文件名：0_0001.parquet
        dst = subfolder / new_name

        if dst.exists() and not OVERWRITE:
            # 防止重复运行导致冲突：自动追加 _1/_2 …
            k = 1
            while dst.exists():
                new_name = f"{vehicle_id}_{k}.parquet"
                dst = subfolder / new_name
                k += 1

        # 复制 parquet
        shutil.copy2(src, dst)

        # 复制标记文件（来自 parquet 源文件所在目录）
        copy_marker_files(src.parent, subfolder)

        # === 仅修改：对照表改为 “hashed_vin（等号后） ↔ vehicle_id” ===
        # 取源文件所在子文件夹名，如：hashed_vin=00a7dede40...
        src_subfolder = src.parent.name
        # 只用等号后面的编码；若无等号则原样返回
        if "=" in src_subfolder:
            hashed_vin = src_subfolder.split("=", 1)[1]
        else:
            hashed_vin = src_subfolder

        rows.append({
            "hashed_vin": hashed_vin,   # 仅等号后的编码
            "vehicle_id": vehicle_id,   # 新的车辆编号（如 0_0001）
        })
        # === 修改结束 ===

        idx += 1

    # === 导出对照表 CSV（两列表：hashed_vin, vehicle_id） ===
    df = pd.DataFrame(rows, columns=["hashed_vin", "vehicle_id"])
    csv_path = out_root / "rename_mapping_with_subfolder.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] 共处理 {len(rows)} 个 parquet 文件。")
    print(f"[DONE] 对照表：{csv_path}")
    print("[TIP] 每个输出子文件夹中已包含其源目录的 _SUCCESS / _started_* / _committed_* 标记文件。")

if __name__ == "__main__":
    main()
