# -*- coding: utf-8 -*-
import re
from pathlib import Path

# === 修改为你的 pkl 文件所在文件夹 ===
DIR = Path(r"C:\Users\YIFSHEN\Documents\01_InputRawData\1000_normalpkl")

KEEP_MIN, KEEP_MAX = 1, 500        # 保留 0_0001 ~ 0_0500（含）
pattern = re.compile(r"0_(\d{4})")  # 从文件名中提取编号
deleted = kept = skipped = 0

if not DIR.exists() or not DIR.is_dir():
    raise SystemExit(f"[ERROR] 目录不存在或不是文件夹: {DIR}")

for p in DIR.glob("*.pkl"):
    m = pattern.search(p.name)
    if not m:
        skipped += 1
        continue
    idx = int(m.group(1))
    if KEEP_MIN <= idx <= KEEP_MAX:
        kept += 1
        continue
    try:
        p.unlink()   # 删除
        deleted += 1
    except Exception as e:
        print(f"[FAIL] 无法删除: {p} -> {e}")

print(f"[DONE] 保留: {kept} 个 (0_{KEEP_MIN:04d}~0_{KEEP_MAX:04d})，"
      f"删除: {deleted} 个，跳过(无匹配编号): {skipped} 个。")
