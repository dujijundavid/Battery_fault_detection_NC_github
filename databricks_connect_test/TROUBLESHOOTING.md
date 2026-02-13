# Databricks Connect 故障排查指南

## 当前问题诊断

### 错误信息
```
java.lang.UnsupportedClassVersionError: org/apache/spark/launcher/Main has been compiled by a more recent version of the Java Runtime (class file version 61.0), this version of the Java Runtime only recognizes class file versions up to 52.0
```

### 问题分析
| 组件 | 当前版本 | 要求版本 | 状态 |
|------|----------|----------|------|
| Java | 1.8 (Java 8) | 17+ | ❌ 不兼容 |
| databricks-connect | 16.4.9 | 需要 Java 17 | ❌ 不兼容 |
| pyspark | 4.0.1 | 需要 Java 17 | ❌ 不兼容 |

**Class file version 对应关系：**
- 52.0 = Java 8
- 61.0 = Java 17

## 解决方案

### 方案 1：升级到 Java 17（推荐）

```bash
# macOS 使用 Homebrew
brew install openjdk@17

# 设置 JAVA_HOME
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 17)' >> ~/.zshrc
source ~/.zshrc

# 验证
java -version
```

然后重新运行测试：
```bash
python simple_gpu_test.py
```

### 方案 2：降级 databricks-connect

如果您无法升级 Java，可以降级到兼容 Java 8 的版本：

```bash
# 卸载当前版本
pip uninstall databricks-connect pyspark

# 安装兼容 Java 8 的版本
pip install databricks-connect==14.1.0 pyspark==3.5.0
```

### 方案 3：使用 Databricks SDK 直接提交 Job（最简单）

不需要本地 Java 兼容性，直接通过 API 提交到集群：

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# 上传并运行 notebook
run = w.jobs.run_now(
    job_id="your-job-id",
    notebook_params={"param": "value"}
)
```

## 在 VSCode 中运行 GPU 代码的最佳实践

### 推荐：使用 Databricks VSCode 扩展

1. 安装 [Databricks VSCode 扩展](https://marketplace.visualstudio.com/items?itemName=databricks.databricks)
2. 配置连接到您的 workspace
3. 直接在 VSCode 中编辑并运行 notebook

### 替代：使用 databricks-cli

```bash
# 将 notebook 上传到 workspace
databricks workspace import ./GPU_test.ipynb /Workspace/Users/your@email.com/GPU_test.ipynb

# 通过 CLI 运行（需要配置好）
```

## 快速验证脚本

使用以下脚本验证您的环境配置：

```bash
# 1. 检查 Java 版本
java -version

# 2. 检查 Python 包版本
pip list | grep -E "pyspark|databricks"

# 3. 测试 Databricks Connect
python -c "from databricks.connect import DatabricksSession; print('OK')"

# 4. 测试 Spark 连接
python databricks_connect_smoke_test.py
```

## 常见问题

### Q: 为什么需要 Java？
A: PySpark 和 Databricks Connect 底层使用 JVM，需要 Java 运行环境。

### Q: 能不能在本地不装 Java 直接运行？
A: 可以使用 Databricks SDK 或 VSCode 扩展，它们通过 REST API 通信，不需要本地 Java。

### Q: 哪种方案最简单？
A: 对于交互式开发，使用 **Databricks VSCode 扩展** 是最简单的选择。

## 下一步

1. 选择一个解决方案（推荐升级 Java 或使用 VSCode 扩展）
2. 重新运行测试脚本
3. 如果成功，运行您的 GPU notebook
