from pyspark.sql import SparkSession

# 初始化SparkSession
spark = SparkSession.builder \
    .appName("ReadParquetTop10") \
    .master("local[*]") \
    .getOrCreate()

# 定义文件路径
parquet_path = '/Volumes/conf/dl/vol_prediction-rdeb-yifshen_common_blob/203_Validation_new/0_3000/'

try:
    # 读取parquet文件
    df = spark.read.parquet(parquet_path)
    # 显示前10行（不截断文本）
    df.show(10, truncate=False)
    # 可选：查看数据结构
    df.printSchema()
except Exception as e:
    print(f"读取失败：{str(e)}")
finally:
    # 释放资源
    spark.stop()