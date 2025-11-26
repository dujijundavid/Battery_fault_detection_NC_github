"""
map_hashed_vin_simple.py
用法（仅三个参数，全是路径）：
python map_hashed_vin_simple.py --map C:\data\mapping.csv --input C:\data\input.csv --output C:\data\matched.csv
要求：
- 对照表 CSV 至少包含列：hashed_vin, vehicle_id
- 输入表 CSV 至少包含列：hashed_vin
- 输出仅包含列：hashed_vin, vehicle_id
"""
import argparse
from pathlib import Path
import sys
import pandas as pd
def clean_hashed_vin_series(s: pd.Series) -> pd.Series:
    """清洗 hashed_vin：去空白、若包含 '=' 取等号后半段、统一转小写。"""
    s = s.astype(str).fillna("").str.strip()
    s = s.str.split("=").str[-1] # 兼容 'hashed_vin=xxxx' 或多重 '='
    return s.str.lower()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=r"C:\Users\YIFSHEN\Documents\01_InputRawData\rename_mapping_with_subfolder.csv", help=r"C:\Users\YIFSHEN\Documents\01_InputRawData\rename_mapping_with_subfolder.csv")
    ap.add_argument("--input", default=r"C:\Users\YIFSHEN\Documents\01_InputRawData\40-70mv_hashed_vin_list.csv", help=r"C:\Users\YIFSHEN\Documents\01_InputRawData\40-70mv_hashed_vin_list.csv")
    ap.add_argument("--output", default=r"C:\Users\YIFSHEN\Documents\01_InputRawData\40-70mv_hashed_vin_list_new.csv", help=r"C:\Users\YIFSHEN\Documents\01_InputRawData\40-70mv_hashed_vin_list_new,csv")
    args = ap.parse_args()
    map_path = Path(args.map)
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not map_path.exists():
        sys.exit(f"[ERROR] 对照表不存在：{map_path}")
    if not in_path.exists():
        sys.exit(f"[ERROR] 输入表不存在：{in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 读取对照表与输入表（全部按字符串，保留前导零）
    try:
        mapping = pd.read_csv(map_path, dtype=str, usecols=["hashed_vin", "vehicle_id"], encoding="utf-8")
    except ValueError:
        sys.exit("[ERROR] 对照表必须包含列：hashed_vin, vehicle_id")
    try:
        df_in = pd.read_csv(in_path, dtype=str, usecols=["hashed_vin"], encoding="utf-8")
    except ValueError:
        sys.exit("[ERROR] 输入表必须包含列：hashed_vin")
    # 清洗
    mapping["hashed_vin"] = clean_hashed_vin_series(mapping["hashed_vin"])
    df_in["hashed_vin"] = clean_hashed_vin_series(df_in["hashed_vin"])
    # 去空、去重
    mapping = mapping.dropna(subset=["hashed_vin"]).drop_duplicates(subset=["hashed_vin"], keep="last")
    df_in = df_in.dropna(subset=["hashed_vin"]).drop_duplicates(subset=["hashed_vin"])
    # 左连接
    result = df_in.merge(mapping, on="hashed_vin", how="left")[["hashed_vin", "vehicle_id"]]
    # 写出
    result.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] 唯一 hashed_vin：{len(df_in)}，写出：{out_path}")
    missing = result["vehicle_id"].isna().sum()
    if missing:
        print(f"[WARN] 有 {missing} 条 hashed_vin 未在对照表匹配到 vehicle_id。")
if __name__ == "__main__":
    main()