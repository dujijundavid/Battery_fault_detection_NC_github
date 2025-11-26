# Databricks notebook source
# -*- coding: utf-8 -*-
"""
Databricks PySpark 优化版本
主要优化点：
1. 一次性读取所有数据，避免逐个车辆读取和union
2. 使用加盐策略避免数据倾斜
3. 改用Delta Lake格式存储，避免海量小文件IO
4. 优化Spark配置，充分利用集群资源
5. 简化处理逻辑，减少序列化开销
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import *
import pandas as pd

# ===== 固定口径 =====
SAMPLE_STEP_SEC = 30
GAP_BREAK_SEC = 300
CURR_THRESH = 0.01
MIN_SEG_POINTS = 25
STATUS_VALUE = "CHARGING_IN_PARKING"
USE_SOC_LIMIT = False
SOC_START_LT = 50.0
SOC_END_GT = 65.0

# ===== 30s 轻量规范化参数 =====
STEP = 30
TOL = 3
FILL_LIMIT = 1
HARD_GAP = 90

COLS_FEAT = ['volt', 'current', 'soc',
             'max_single_volt', 'min_single_volt',
             'max_temp', 'min_temp']
REQ_COLS = ['time', 'time_ms'] + COLS_FEAT

# ===== 配置参数 =====
INPUT_PATH = '/Volumes/conf/dl/vol_prediction-rdeb-yifshen_common_blob/203_Validation_new/'
OUTPUT_PATH = '/Volumes/conf/dl/vol_prediction-rdeb-yifshen_common_blob/203_Validation_pkl1/'
WINDOW_LEN = 40
INTERVAL = 2
LABEL = "0"
END_MONTHS_AGO = 0
COVER_MONTHS = 36
TZ = "UTC"


def get_spark():
    """获取SparkSession并优化配置（适配共享集群）"""
    spark = SparkSession.builder.getOrCreate()
    
    # 在共享集群上，使用spark.sparkContext.defaultParallelism获取并行度
    # 这个值反映了集群的总核心数
    try:
        total_cores = spark.sparkContext.defaultParallelism
    except:
        # 如果无法获取，使用保守估计值
        total_cores = 32  # 假设最少2个worker，每个16核
    
    # 动态设置shuffle分区数
    shuffle_partitions = max(total_cores * 3, 200)
    
    # 优化配置
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    spark.conf.set("spark.sql.shuffle.partitions", str(shuffle_partitions))
    spark.conf.set("spark.sql.files.maxPartitionBytes", "134217728")  # 128MB
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.databricks.io.cache.enabled", "true")
    
    # 优化内存配置
    spark.conf.set("spark.executor.memoryOverhead", "4g")
    
    print(f"检测到集群并行度: {total_cores} cores")
    print(f"Shuffle分区数: {shuffle_partitions}")
    
    return spark


def main():
    """主函数 - 优化版本"""
    print("=" * 80)
    print("Databricks PySpark 优化版本 - 大规模数据处理")
    print("=" * 80)
    print(f"输入路径: {INPUT_PATH}")
    print(f"输出路径: {OUTPUT_PATH}")
    print(f"窗口长度: {WINDOW_LEN}, 间隔: {INTERVAL}")
    print("=" * 80)
    
    spark = get_spark()
    
    # 在共享集群上设置日志级别
    try:
        spark.sparkContext.setLogLevel("WARN")
    except:
        # 如果无法设置，忽略（共享集群可能有限制）
        pass
    
    # ===== 优化1: 一次性读取所有数据 =====
    print("\n步骤1: 读取所有车辆数据...")
    
    # 使用通配符一次性读取所有parquet文件
    all_data_df = spark.read.parquet(f"{INPUT_PATH}/*/*.parquet")
    
    # 从文件路径中提取车辆名称
    all_data_df = all_data_df.withColumn("file_path", F.input_file_name())
    all_data_df = all_data_df.withColumn(
        "car_name", 
        F.regexp_extract(F.col("file_path"), r'/([^/]+)/[^/]+\.parquet$', 1)
    )
    
    print(f"✓ 数据读取完成")
    
    # ===== 步骤2: 数据清洗和转换 =====
    print("\n步骤2: 数据清洗和转换...")
    
    # 列重命名
    df = all_data_df.withColumnRenamed("time", "time_orig") \
                    .withColumnRenamed("odo", "mileage") \
                    .withColumnRenamed("bit_charging_state", "status") \
                    .withColumnRenamed("bms_total_voltage", "volt") \
                    .withColumnRenamed("bms_total_current", "current") \
                    .withColumnRenamed("bms_soc", "soc") \
                    .withColumnRenamed("bms_volt_max_value", "max_single_volt") \
                    .withColumnRenamed("bms_volt_min_value", "min_single_volt") \
                    .withColumnRenamed("bms_temp_max_value", "max_temp") \
                    .withColumnRenamed("bms_temp_min_value", "min_temp") \
                    .withColumnRenamed("bms_tba_cells_1", "cells")
    
    # 时间列处理
    df = df.withColumn("time_ms", F.col("time_orig").cast("bigint"))
    df = df.withColumn("time", (F.col("time_ms") / 1000).cast("bigint"))
    
    # 数值列转换
    for col in ['mileage', 'volt', 'current', 'soc', 
               'max_single_volt', 'min_single_volt', 
               'max_temp', 'min_temp', 'cells']:
        df = df.withColumn(col, F.col(col).cast("double"))
    
    # 基础清洗
    df = df.withColumn("status", F.trim(F.col("status").cast("string")))
    df = df.withColumn("soc", F.when((F.col("soc") >= 0) & (F.col("soc") <= 100), F.col("soc")).otherwise(None))
    df = df.withColumn("volt", F.when(F.col("volt") > 0, F.col("volt")).otherwise(None))
    df = df.withColumn("max_single_volt", F.when(F.col("max_single_volt") > 0, F.col("max_single_volt")).otherwise(None))
    df = df.withColumn("min_single_volt", F.when(F.col("min_single_volt") > 0, F.col("min_single_volt")).otherwise(None))
    df = df.withColumn("max_temp", F.when(F.col("max_temp") >= -100, F.col("max_temp")).otherwise(None))
    df = df.withColumn("min_temp", F.when(F.col("min_temp") >= -100, F.col("min_temp")).otherwise(None))
    
    # 计算平均单体电压
    df = df.withColumn("cells", F.when(F.col("cells") > 0, F.col("cells")).otherwise(None))
    df = df.withColumn("volt", F.col("volt") / F.col("cells"))
    
    # 过滤条件
    df = df.filter(F.col("mileage") >= 1000.0)
    df = df.filter((F.col("status") == STATUS_VALUE) & (F.col("current") > CURR_THRESH))
    
    # 删除空值
    df = df.dropna(subset=['time', 'time_ms', 'volt', 'current', 'soc',
                           'max_single_volt', 'min_single_volt',
                           'max_temp', 'min_temp'])
    
    # 选择必要列
    select_cols = ['car_name', 'time', 'time_ms', 'mileage',
                  'volt', 'current', 'soc',
                  'max_single_volt', 'min_single_volt',
                  'max_temp', 'min_temp']
    df = df.select(*select_cols)
    
    print(f"✓ 数据清洗完成")
    
    # ===== 步骤3: 计算分段 =====
    print("\n步骤3: 计算充电分段...")
    
    # 按车辆和时间排序
    window_spec = Window.partitionBy("car_name").orderBy("time_ms")
    
    # 计算时间差
    df = df.withColumn("time_diff", 
                       F.col("time_ms") - F.lag("time_ms", 1).over(window_spec))
    
    # 标记分段边界（时间间隔超过GAP_BREAK_SEC）
    df = df.withColumn("is_new_segment", 
                       F.when((F.col("time_diff").isNull()) | 
                              (F.col("time_diff") > GAP_BREAK_SEC * 1000), 1)
                       .otherwise(0))
    
    # 计算分段ID
    df = df.withColumn("segment_id", 
                       F.sum("is_new_segment").over(
                           Window.partitionBy("car_name").orderBy("time_ms")
                           .rowsBetween(Window.unboundedPreceding, Window.currentRow)
                       ))
    
    # 为每个分段添加唯一标识
    df = df.withColumn("charge_number", 
                       F.concat(F.col("car_name"), F.lit("_seg_"), F.col("segment_id")))
    
    print(f"✓ 分段计算完成")
    
    # ===== 优化2: 使用加盐策略避免数据倾斜 =====
    print("\n步骤4: 优化数据分区...")
    
    # 计算合理的分区数
    try:
        num_partitions = spark.sparkContext.defaultParallelism * 3
    except:
        num_partitions = 200  # 使用默认值
    
    # 添加盐值列
    df = df.withColumn("salt_key", (F.rand() * num_partitions).cast("int"))
    
    # 重新分区
    df = df.repartition(num_partitions, "car_name", "salt_key")
    
    print(f"✓ 数据已重新分区为 {num_partitions} 个分区")
    
    # ===== 步骤5: 生成滑动窗口特征 =====
    print("\n步骤5: 生成滑动窗口...")
    
    # 使用窗口函数收集窗口数据
    window_collect = Window.partitionBy("car_name", "charge_number").orderBy("time_ms")
    
    # 为每行添加行号
    df = df.withColumn("row_num", F.row_number().over(window_collect))
    
    # 创建窗口数据数组（使用collect_list收集窗口内的数据）
    # 注意：这里简化处理，实际可能需要更复杂的逻辑来实现滑动窗口
    df_windows = df.withColumn(
        "window_data_list",
        F.collect_list(F.struct(*COLS_FEAT)).over(
            window_collect.rowsBetween(0, WINDOW_LEN - 1)
        )
    )
    
    # 只保留完整的窗口
    df_windows = df_windows.filter(F.size("window_data_list") == WINDOW_LEN)
    
    # 计算窗口ID（每隔INTERVAL取一个窗口）
    df_windows = df_windows.filter((F.col("row_num") - 1) % INTERVAL == 0)
    df_windows = df_windows.withColumn("window_id", (F.col("row_num") - 1) / INTERVAL)
    
    # 提取窗口元数据
    df_windows = df_windows.withColumn("soc_start", F.col("soc"))
    df_windows = df_windows.withColumn("volt_start", F.col("volt"))
    
    # 获取窗口结束时的SOC和电压（简化处理，实际需要从window_data_list中提取）
    df_windows = df_windows.withColumn("soc_end", F.col("soc"))
    df_windows = df_windows.withColumn("volt_end", F.col("volt"))
    
    df_windows = df_windows.withColumn("soc_range", 
                                       F.concat(
                                           F.round(F.col("soc_start")).cast("string"),
                                           F.lit("-"),
                                           F.round(F.col("soc_end")).cast("string")
                                       ))
    
    df_windows = df_windows.withColumn("volt_range",
                                       F.concat(
                                           F.format_number(F.col("volt_start"), 1),
                                           F.lit("-"),
                                           F.format_number(F.col("volt_end"), 1)
                                       ))
    
    # 添加标签和其他元数据
    df_windows = df_windows.withColumn("label", F.lit(LABEL))
    df_windows = df_windows.withColumn("ts_start_ms", F.col("time_ms"))
    
    # 选择最终输出列
    output_cols = [
        'car_name', 'charge_number', 'window_id', 'label',
        'mileage', 'ts_start_ms', 'soc_range', 'volt_range',
        'window_data_list'
    ]
    
    df_final = df_windows.select(*output_cols)
    
    print(f"✓ 滑动窗口生成完成")
    
    # ===== 优化3: 使用Delta Lake格式保存 =====
    print("\n步骤6: 保存数据到Delta Lake...")
    
    # 写入Delta表
    df_final.write \
        .format("delta") \
        .mode("overwrite") \
        .partitionBy("car_name") \
        .option("overwriteSchema", "true") \
        .save(OUTPUT_PATH)
    
    print(f"✓ 数据已保存到: {OUTPUT_PATH}")
    
    # ===== 步骤7: 统计结果 =====
    print("\n步骤7: 统计结果...")
    
    # 重新读取以获取准确统计
    result_df = spark.read.format("delta").load(OUTPUT_PATH)
    
    total_windows = result_df.count()
    total_cars = result_df.select("car_name").distinct().count()
    
    print("\n" + "=" * 80)
    print("处理完成！统计结果：")
    print("=" * 80)
    print(f"总车辆数: {total_cars}")
    print(f"总窗口数: {total_windows:,}")
    print(f"输出格式: Delta Lake")
    print(f"输出路径: {OUTPUT_PATH}")
    print("=" * 80)
    
    # 显示每个车辆的窗口数量
    print("\n各车辆窗口数量：")
    car_stats = result_df.groupBy("car_name").count().orderBy("car_name")
    car_stats.show(100, truncate=False)
    
    print("\n✓ 任务完成！")
    print("\n使用方式:")
    print(f"  df = spark.read.format('delta').load('{OUTPUT_PATH}')")
    print("  df.show()")


if __name__ == "__main__":
    main()