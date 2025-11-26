#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Battery Data Processing Pipeline
=================================

Unified script for processing battery charging data using PySpark.
This script replaces multiple legacy processing scripts with a single,
configuration-driven approach.

Usage:
    spark-submit data_process.py [--config config.yaml]

Author: Refactored based on ML Engineer review
Date: 2025-11-26
"""

import os
import sys
import argparse
import yaml
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import *


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_spark_session(config):
    """Create and configure SparkSession."""
    spark_config = config.get('spark', {})
    
    builder = SparkSession.builder \
        .appName(spark_config.get('app_name', 'BatteryDataProcessing'))
    
    # Get existing session or create new one
    spark = builder.getOrCreate()
    
    # Apply configuration
    spark.conf.set("spark.sql.shuffle.partitions", 
                   str(spark_config.get('shuffle_partitions', 200)))
    spark.conf.set("spark.sql.adaptive.enabled", 
                   str(spark_config.get('adaptive_enabled', True)).lower())
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    spark.conf.set("spark.sql.files.maxPartitionBytes", "134217728")  # 128MB
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    
    # Databricks specific
    if spark_config.get('io_cache_enabled', True):
        spark.conf.set("spark.databricks.io.cache.enabled", "true")
    
    if spark_config.get('memory_overhead'):
        spark.conf.set("spark.executor.memoryOverhead", 
                      spark_config.get('memory_overhead'))
    
    # Set log level
    try:
        spark.sparkContext.setLogLevel("WARN")
    except:
        pass
    
    return spark


def read_raw_data(spark, config):
    """Read raw parquet data from input directory."""
    input_path = config['data']['input']
    
    print(f"\n{'='*80}")
    print("Step 1: Reading raw data...")
    print(f"Input path: {input_path}")
    
    # Read all parquet files at once
    df = spark.read.parquet(f"{input_path}/*/*.parquet")
    
    # Extract car name from file path
    df = df.withColumn("file_path", F.input_file_name())
    df = df.withColumn(
        "car_name",
        F.regexp_extract(F.col("file_path"), r'/([^/]+)/[^/]+\.parquet$', 1)
    )
    
    # Apply car whitelist if specified
    car_whitelist = config['dataset'].get('car_whitelist', '')
    if car_whitelist:
        print(f"Applying car whitelist: {car_whitelist}")
        # Simple comma-separated or regex
        if ',' in car_whitelist:
            cars = [c.strip() for c in car_whitelist.split(',') if c.strip()]
            df = df.filter(F.col("car_name").isin(cars))
        else:
            df = df.filter(F.col("car_name").rlike(car_whitelist))
    
    print(f"✓ Data loaded successfully")
    return df


def clean_data(df, config):
    """Clean and transform raw data."""
    print("\n{'='*80}")
    print("Step 2: Cleaning and transforming data...")
    
    filters = config['features']['filters']
    
    # Column renaming
    df = df \
        .withColumnRenamed("time", "time_orig") \
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
    
    # Time column processing
    df = df.withColumn("time_ms", F.col("time_orig").cast("bigint"))
    df = df.withColumn("time", (F.col("time_ms") / 1000).cast("bigint"))
    
    # Convert numerical columns
    for col in ['mileage', 'volt', 'current', 'soc',
                'max_single_volt', 'min_single_volt',
                'max_temp', 'min_temp', 'cells']:
        df = df.withColumn(col, F.col(col).cast("double"))
    
    # Clean status column
    df = df.withColumn("status", F.trim(F.col("status").cast("string")))
    
    # Apply data quality filters
    soc_range = filters.get('soc_range', [0, 100])
    df = df.withColumn("soc", 
                       F.when(F.col("soc").between(soc_range[0], soc_range[1]), 
                              F.col("soc")).otherwise(None))
    
    volt_min = filters.get('voltage_min', 0)
    df = df.withColumn("volt", 
                       F.when(F.col("volt") > volt_min, F.col("volt")).otherwise(None))
    df = df.withColumn("max_single_volt",
                       F.when(F.col("max_single_volt") > volt_min, 
                              F.col("max_single_volt")).otherwise(None))
    df = df.withColumn("min_single_volt",
                       F.when(F.col("min_single_volt") > volt_min, 
                              F.col("min_single_volt")).otherwise(None))
    
    temp_min = filters.get('temp_min', -100)
    df = df.withColumn("max_temp",
                       F.when(F.col("max_temp") >= temp_min, 
                              F.col("max_temp")).otherwise(None))
    df = df.withColumn("min_temp",
                       F.when(F.col("min_temp") >= temp_min, 
                              F.col("min_temp")).otherwise(None))
    
    # Calculate average cell voltage
    df = df.withColumn("cells", 
                       F.when(F.col("cells") > 0, F.col("cells")).otherwise(None))
    df = df.withColumn("volt", F.col("volt") / F.col("cells"))
    
    # Apply mileage filter
    min_mileage = filters.get('min_mileage', 1000.0)
    df = df.filter(F.col("mileage") >= min_mileage)
    
    # Apply charging filters
    charging_cfg = config['features']['charging']
    df = df.filter(
        (F.col("status") == charging_cfg['status']) &
        (F.col("current") > charging_cfg['current_threshold'])
    )
    
    # Drop rows with missing critical values
    df = df.dropna(subset=[
        'time', 'time_ms', 'volt', 'current', 'soc',
        'max_single_volt', 'min_single_volt',
        'max_temp', 'min_temp'
    ])
    
    # Select final columns
    select_cols = [
        'car_name', 'time', 'time_ms', 'mileage',
        'volt', 'current', 'soc',
        'max_single_volt', 'min_single_volt',
        'max_temp', 'min_temp'
    ]
    df = df.select(*select_cols)
    
    print(f"✓ Data cleaning complete")
    return df


def identify_charging_segments(df, config):
    """Identify charging segments using time gaps."""
    print("\n{'='*80}")
    print("Step 3: Identifying charging segments...")
    
    charging_cfg = config['features']['charging']
    gap_threshold = charging_cfg['gap_threshold']
    min_points = charging_cfg['min_points']
    
    # Create window spec for each car
    window_spec = Window.partitionBy("car_name").orderBy("time_ms")
    
    # Calculate time difference from previous record
    df = df.withColumn("time_diff",
                       F.col("time_ms") - F.lag("time_ms", 1).over(window_spec))
    
    # Mark segment boundaries (gap > threshold or first record)
    df = df.withColumn("is_new_segment",
                       F.when((F.col("time_diff").isNull()) |
                              (F.col("time_diff") > gap_threshold * 1000), 1)
                       .otherwise(0))
    
    # Assign segment IDs
    df = df.withColumn("segment_id",
                       F.sum("is_new_segment").over(
                           Window.partitionBy("car_name").orderBy("time_ms")
                           .rowsBetween(Window.unboundedPreceding, Window.currentRow)
                       ))
    
    # Create unique charge session identifier
    df = df.withColumn("charge_number",
                       F.concat(F.col("car_name"), F.lit("_seg_"), F.col("segment_id")))
    
    # Calculate segment sizes for filtering
    segment_size_df = df.groupBy("car_name", "charge_number").count()
    segment_size_df = segment_size_df.filter(F.col("count") >= min_points)
    
    # Filter out small segments
    df = df.join(
        segment_size_df.select("car_name", "charge_number"),
        on=["car_name", "charge_number"],
        how="inner"
    )
    
    print(f"✓ Charging segments identified (min {min_points} points)")
    return df


def create_sliding_windows(df, config):
    """Generate sliding windows from charging segments."""
    print("\n{'='*80}")
    print("Step 4: Creating sliding windows...")
    
    window_len = config['features']['window_length']
    interval = config['features']['window_interval']
    label = config['dataset']['label']
    
    # Feature columns
    feat_cols = ['volt', 'current', 'soc',
                 'max_single_volt', 'min_single_volt',
                 'max_temp', 'min_temp']
    
    # Window spec for collecting features
    window_collect = Window.partitionBy("car_name", "charge_number").orderBy("time_ms")
    
    # Add row number within each segment
    df = df.withColumn("row_num", F.row_number().over(window_collect))
    
    # Collect sliding window data
    df_windows = df.withColumn(
        "window_data",
        F.collect_list(F.struct(*feat_cols)).over(
            window_collect.rowsBetween(0, window_len - 1)
        )
    )
    
    # Keep only complete windows
    df_windows = df_windows.filter(F.size("window_data") == window_len)
    
    # Apply interval stride
    df_windows = df_windows.filter((F.col("row_num") - 1) % interval == 0)
    
    # Add window ID
    df_windows = df_windows.withColumn("window_id", 
                                       ((F.col("row_num") - 1) / interval).cast("int"))
    
    # Extract window metadata
    df_windows = df_windows.withColumn("soc_start", F.col("soc"))
    df_windows = df_windows.withColumn("volt_start", F.col("volt"))
    
    # Get end values from the window
    df_windows = df_windows.withColumn(
        "soc_end",
        F.element_at(F.col("window_data").getField("soc"), -1)
    )
    df_windows = df_windows.withColumn(
        "volt_end",
        F.element_at(F.col("window_data").getField("volt"), -1)
    )
    
    # Create range strings
    df_windows = df_windows.withColumn(
        "soc_range",
        F.concat(
            F.round(F.col("soc_start")).cast("string"),
            F.lit("-"),
            F.round(F.col("soc_end")).cast("string")
        )
    )
    
    df_windows = df_windows.withColumn(
        "volt_range",
        F.concat(
            F.format_number(F.col("volt_start"), 1),
            F.lit("-"),
            F.format_number(F.col("volt_end"), 1)
        )
    )
    
    # Add label and timestamp
    df_windows = df_windows.withColumn("label", F.lit(label))
    df_windows = df_windows.withColumn("ts_start_ms", F.col("time_ms"))
    
    # Select final output columns
    output_cols = [
        'car_name', 'charge_number', 'window_id', 'label',
        'mileage', 'ts_start_ms', 'soc_range', 'volt_range',
        'window_data'
    ]
    
    df_final = df_windows.select(*output_cols)
    
    print(f"✓ Sliding windows created (length={window_len}, interval={interval})")
    return df_final


def save_results(df, config):
    """Save processed data to output location."""
    print("\n{'='*80}")
    print("Step 5: Saving results...")
    
    output_path = config['data']['output']
    output_config = config['output']
    
    # Write data
    writer = df.write \
        .format(output_config.get('format', 'delta')) \
        .mode(output_config.get('mode', 'overwrite'))
    
    # Apply partitioning if specified
    partition_by = output_config.get('partition_by')
    if partition_by:
        writer = writer.partitionBy(partition_by)
    
    writer.option("overwriteSchema", "true") \
          .save(output_path)
    
    print(f"✓ Data saved to: {output_path}")
    return output_path


def print_statistics(spark, output_path, config):
    """Print processing statistics."""
    print("\n{'='*80}")
    print("Step 6: Computing statistics...")
    
    # Read the output data
    output_format = config['output'].get('format', 'delta')
    df = spark.read.format(output_format).load(output_path)
    
    # Compute statistics
    total_windows = df.count()
    total_cars = df.select("car_name").distinct().count()
    
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total vehicles processed: {total_cars}")
    print(f"Total windows generated: {total_windows:,}")
    print(f"Output format: {output_format}")
    print(f"Output location: {output_path}")
    print(f"{'='*80}")
    
    # Show per-vehicle statistics
    print("\nWindows per vehicle:")
    car_stats = df.groupBy("car_name") \
        .count() \
        .orderBy("car_name")
    car_stats.show(100, truncate=False)
    
    print(f"\n✓ All done!")
    print(f"\nTo read the data:")
    print(f"  df = spark.read.format('{output_format}').load('{output_path}')")
    print(f"  df.show()")


def main():
    """Main execution pipeline."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Battery data processing pipeline using PySpark'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()
    
    # Load configuration
    print(f"\n{'='*80}")
    print("BATTERY DATA PROCESSING PIPELINE")
    print(f"{'='*80}")
    print(f"Loading configuration from: {args.config}")
    
    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    print(f"✓ Configuration loaded")
    
    # Initialize Spark
    spark = get_spark_session(config)
    print(f"✓ Spark session initialized")
    
    # Execute pipeline
    try:
        df = read_raw_data(spark, config)
        df = clean_data(df, config)
        df = identify_charging_segments(df, config)
        df = create_sliding_windows(df, config)
        output_path = save_results(df, config)
        print_statistics(spark, output_path, config)
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Processing failed!")
        print(f"{'='*80}")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
