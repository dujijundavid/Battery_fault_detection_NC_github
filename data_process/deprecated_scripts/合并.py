import os
import csv
import time
import threading
from queue import Queue
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# 初始化SparkSession（优化集群参数）
spark = SparkSession.builder \
    .appName("ParquetMerger") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.instances", "6") \
    .config("spark.default.parallelism", "96") \
    .config("spark.sql.shuffle.partitions", "96") \
    .getOrCreate()

# 路径配置
source_root = '/Volumes/conf/dl/vol_prediction-rdeb-yifshen_common_blob/203_Validation/'
target_root = '/Volumes/conf/dl/vol_prediction-rdeb-yifshen_common_blob/203_Validation_new/'
mapping_path = '/Workspace/Users/pssx_a_yifshen@corpdir.partner.onmschina.cn/VAE/data_process/folder_mapping.csv'

# 创建目标目录
os.makedirs(target_root, exist_ok=True)
os.makedirs(os.path.dirname(mapping_path), exist_ok=True)

# 获取所有子文件夹（按名称排序确保映射稳定）
subfolders = sorted([
    f for f in os.listdir(source_root) 
    if os.path.isdir(os.path.join(source_root, f))
])
total_folders = len(subfolders)
print(f"发现 {total_folders} 个子文件夹需要处理")

# 生成文件夹映射关系（0_3000开始）
folder_mapping = {}
start_num = 3000
for i, old_name in enumerate(subfolders):
    new_name = f"0_{start_num + i}"
    folder_mapping[old_name] = new_name

# 保存映射关系到CSV
with open(mapping_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['original_folder', 'new_folder'])
    for old, new in folder_mapping.items():
        writer.writerow([old, new])
print(f"文件夹映射关系已保存至: {mapping_path}")

# 进度跟踪变量
processed_count = 0
lock = threading.Lock()

def update_progress():
    """更新并显示进度条"""
    global processed_count
    with lock:
        processed_count += 1
        progress = (processed_count / total_folders) * 100
        bar_length = 50
        filled_length = int(bar_length * progress // 100)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        print(f"\r进度: [{bar}] {progress:.2f}% ({processed_count}/{total_folders})", end='', flush=True)

def process_folder(old_folder, new_folder):
    """处理单个文件夹：合并Parquet为单个文件并排序"""
    try:
        # 读取文件夹内所有Parquet文件
        source_path = os.path.join(source_root, old_folder)
        df = spark.read.parquet(source_path)
        
        # 验证time列存在
        if 'time' not in df.columns:
            raise ValueError(f"文件夹 {old_folder} 缺少 'time' 列")
        
        # 按time列排序（升序）并合并为单个分区
        sorted_df = df.orderBy(col('time').asc()).coalesce(1)
        
        # 目标路径结构：新文件夹/新文件夹.parquet（单个文件）
        target_folder = os.path.join(target_root, new_folder)
        os.makedirs(target_folder, exist_ok=True)  # 创建新文件夹
        temp_output_path = os.path.join(target_folder, "temp")  # 临时输出路径
        final_output_path = os.path.join(target_folder, f"{new_folder}.parquet")  # 最终文件路径
        
        # 写入临时目录
        sorted_df.write.mode("overwrite").parquet(temp_output_path)
        
        # 找到临时目录中生成的Parquet文件（part-xxxx.snappy.parquet）
        parquet_files = [
            f for f in os.listdir(temp_output_path) 
            if f.endswith(".parquet") and f.startswith("part-")
        ]
        if not parquet_files:
            raise FileNotFoundError(f"合并后未生成Parquet文件: {temp_output_path}")
        
        # 将临时文件重命名为目标文件（移除Spark生成的碎片名称）
        temp_parquet_path = os.path.join(temp_output_path, parquet_files[0])
        os.rename(temp_parquet_path, final_output_path)
        
        # 清理临时目录
        for f in os.listdir(temp_output_path):
            os.remove(os.path.join(temp_output_path, f))
        os.rmdir(temp_output_path)
        
        update_progress()
        
    except Exception as e:
        with lock:
            print(f"\n处理文件夹 {old_folder} 时出错: {str(e)}")

def worker(queue):
    """线程工作函数"""
    while not queue.empty():
        old_folder, new_folder = queue.get()
        try:
            process_folder(old_folder, new_folder)
        finally:
            queue.task_done()

# 创建任务队列
queue = Queue()
for old, new in folder_mapping.items():
    queue.put((old, new))

# 启动多线程处理（与worker数量匹配）
start_time = time.time()
threads = []
for _ in range(24):  # 使用24个线程，匹配集群worker数
    t = threading.Thread(target=worker, args=(queue,))
    t.daemon = True
    t.start()
    threads.append(t)

# 等待所有任务完成
queue.join()

# 完成信息
end_time = time.time()
print(f"\n所有文件夹处理完成！总耗时: {end_time - start_time:.2f}秒")
spark.stop()