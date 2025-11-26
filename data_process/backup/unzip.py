# Databricks notebook source
# MAGIC %md
# MAGIC # Parquet 格式的文件，并且使用了 Snappy 压缩算法。
# MAGIC # Parquet 是一种列式存储格式，常用于大数据处理场景，它具有高效的压缩率和查询性能，广泛应用于 Apache Hadoop、Spark 等大数据生态系统中。

# COMMAND ----------

import pandas as pd
file_path = r"F:\BATX周期智能\！项目\奔驰\样本数据\MB_sample_data\partitioned_file_name=00be624b-c06c-415d-8afa-b7b07818eecd\part-00228-tid-7990004275262088105-9531764a-9b90-47ad-a8db-3c6c2f7628e8-92707-1.c000.snappy.parquet"
df = pd.read_parquet(file_path) 
print("10：")
print(df.head(10))

# COMMAND ----------

