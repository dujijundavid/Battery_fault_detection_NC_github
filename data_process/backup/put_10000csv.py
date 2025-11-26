import pandas as pd
import os
# 输入的CSV文件路径
input_file_path = r"E:\1研究生阶段资料\10项目\3奔驰项目\MB_sample_data\partitioned_file_name=00be624b-c06c-415d-8afa-b7b07818eecd\sorted_by_time.csv"
# 输出的新CSV文件路径
output_file_path = r"E:\1研究生阶段资料\10项目\3奔驰项目\MB_sample_data\partitioned_file_name=00be624b-c06c-415d-8afa-b7b07818eecd\sorted_first_50000.csv"

try:
    # 读取CSV文件的前10000条数据
    df = pd.read_csv(input_file_path, nrows=50000)

    # 将前10000条数据输出为新CSV文件
    df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

    print(f"导出OK")

    # 输出CSV文件的大小
    file_size_bytes = os.path.getsize(output_file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    print(f"CSV文件大小: {file_size_bytes} byte ({file_size_mb:.2f} MB)")

except Exception as e:
    print(f"出现错误: {e}")
