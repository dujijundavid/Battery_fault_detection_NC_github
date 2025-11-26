import pandas as pd
# 读取现有的 Parquet 文件
parquet_file_path = r"C:\Users\YIFSHEN\Documents\VAE\DyAD\dyad_vae_save\2025-11-12-01-54-20_fold0\result\test_segment_scores.parquet" # 请替换为你文件的路径
df = pd.read_parquet(parquet_file_path)
# 查看前几行数据，确认格式
print(df.head()) # 打印前几行数据
print(df.info()) # 打印数据的详细信息，包括列名、数据类型等
# 可选：查看所有列名
print("Columns in the Parquet file:", df.columns)