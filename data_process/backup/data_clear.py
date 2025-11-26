import pandas as pd
import os

source_file_path = r"C:\Users\YIFSHEN\Documents\01_InputRawData\20_Ti_case_v1_new\1_0001\1_0001.parquet"
output_dir = os.path.dirname(source_file_path)
output_file = os.path.join(output_dir, "sorted_by_time_first_100.csv")

try:
    # 读取Parquet文件
    df = pd.read_parquet(source_file_path)

    # 检查time列
    if 'time' not in df.columns:
        raise ValueError("字段有误")

    # 将time列转换为数值型数据，防止无效数据
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    if df['time'].isna().any():
        print("部分time时间格式有误")
        df = df.dropna(subset=['time'])

    # 对数据按time列进行排序
    df_sorted = df.sort_values(by='time', ascending=True)

    # 只提取前100条数据
    df_first_100 = df_sorted.head(100)

    # 将前100条数据输出为CSV文件
    df_first_100.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"导出OK")

    # 输出CSV文件的大小
    file_size_bytes = os.path.getsize(output_file)
    file_size_mb = file_size_bytes / (1024 * 1024)

    print(f"CSV文件大小: {file_size_bytes} byte ({file_size_mb:.2f} MB)")

except Exception as e:
    print(f"出现错误: {e}")
