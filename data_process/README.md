# Battery Data Processing

Unified PySpark script for processing battery charging data into sliding window features for anomaly detection.

## Quick Start

### 1. Configure Parameters

Edit `config.yaml` to set your data paths and parameters:

```yaml
data:
  input: /path/to/your/input/data/
  output: /path/to/your/output/data/

dataset:
  type: test  # or "train"
  label: "0"  # "0" for normal, "1" for anomaly
```

### 2. Run Processing

```bash
# Local execution (for testing)
spark-submit --master local[4] data_process.py

# Cluster execution (Databricks/YARN)
spark-submit --master yarn --deploy-mode cluster data_process.py

# With custom config file
spark-submit data_process.py --config my_config.yaml
```

### 3. Access Results

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.read.format("delta").load("/path/to/output")
df.show()
```

## Configuration Guide

All parameters are in `config.yaml`:

| Section    | Key                  | Description                                   | Default  |
| ---------- | -------------------- | --------------------------------------------- | -------- |
| `data`     | `input`              | Input directory with vehicle parquet files    | Required |
| `data`     | `output`             | Output directory for processed data           | Required |
| `dataset`  | `type`               | Dataset type: "train" or "test"               | test     |
| `dataset`  | `label`              | Label for data: "0" (normal) or "1" (anomaly) | "0"      |
| `features` | `window_length`      | Number of time steps per window               | 40       |
| `features` | `window_interval`    | Stride between windows                        | 2        |
| `spark`    | `shuffle_partitions` | Number of shuffle partitions                  | 200      |

See `config.yaml` for complete parameter documentation.

## What This Script Does

1. **Reads** battery data from parquet files
2. **Cleans** data (filters invalid values, normalizes columns)
3. **Segments** charging sessions (splits by time gaps)
4. **Windows** creates sliding windows for ML features
5. **Saves** results in Delta Lake format

## Output Schema

```
car_name: string              # Vehicle identifier
charge_number: string         # Unique charging session ID
window_id: int                # Window index within session
label: string                 # Dataset label ("0" or "1")
mileage: double               # Vehicle mileage at window start
ts_start_ms: bigint          # Timestamp of window start (ms)
soc_range: string            # SOC range (e.g., "45-52")
volt_range: string           # Voltage range (e.g., "3.8-3.9")
window_data: array<struct>   # Time series features (40 timesteps)
                             # - volt, current, soc,
                             # - max_single_volt, min_single_volt,
                             # - max_temp, min_temp
```

## Common Issues

### Q: Script is too slow?
**A:** Increase `spark.shuffle_partitions` in `config.yaml`:
```yaml
spark:
  shuffle_partitions: 400  # Increase from default 200
```

### Q: Out of memory errors?
**A:** Increase executor memory:
```yaml
spark:
  executor_memory: 8g  # Increase from default 4g
  memory_overhead: 8g
```

### Q: Want to process specific vehicles only?
**A:** Use `car_whitelist` in config:
```yaml
dataset:
  car_whitelist: "car1,car2,car3"  # Comma-separated
  # OR
  car_whitelist: ".*_3[0-9]{3}"    # Regex pattern
```

### Q: Need to explore data first?
**A:** Use the Jupyter notebook:
```bash
jupyter notebook notebooks/explore_data.ipynb
```

### Q: Want pkl files instead of Delta Lake?
**A:** The script only outputs Delta Lake (recommended). To export as pkl, use:
```python
# In Jupyter or another script
import torch
df = spark.read.format("delta").load("/path/to/output")
data = df.collect()

for row in data:
    window_data = row['window_data']
    metadata = {
        'car': row['car_name'],
        'label': row['label'],
        'mileage': row['mileage'],
        # ... other metadata
    }
    torch.save((window_data, metadata), f"{row['car_name']}_{row['window_id']}.pkl")
```

## Legacy Scripts

Old scripts have been moved to `deprecated_scripts/` for reference. They are no longer maintained.

## File Structure

```
data_process/
├── config.yaml              # Configuration file
├── data_process.py          # Main processing script (this!)
├── README.md                # This file
├── folder_mapping.csv       # Optional: vehicle mapping
├── notebooks/               # Optional: data exploration
│   └── explore_data.ipynb
└── deprecated_scripts/      # Legacy scripts (archived)
    ├── test_data_process_spark.py
    ├── test_data_process.py
    └── ... (8 other old scripts)
```

## Need Help?

1. Check configuration in `config.yaml`
2. Look at common issues above
3. Review Spark logs for detailed error messages
4. Explore data using `notebooks/explore_data.ipynb`

---

**Remember:** This script is designed to be simple and maintainable. If you need custom behavior, modify `config.yaml` first before changing the code.
