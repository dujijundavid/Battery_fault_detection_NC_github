# 配置 Databricks Connect 使用 GPU 集群

## 问题
当前 Databricks Connect 连接的不是您的 GPU 集群，因此 PyTorch 检测不到 CUDA。

## 解决步骤

### 1. 找到您的 GPU 集群 ID

在 Databricks 工作区中：
1. 进入 **Compute** (计算)
2. 找到您的 GPU 集群
3. 点击集群名称进入详情页
4. 从 URL 中复制集群 ID，格式类似：`0123-456789-hopi345`

或者使用 JSON 配置查看：
```
https://adb-xxx.azuredatabricks.net/#setting/companies/xxx/clusters/0123-456789-hopi345/configuration
                                                           ^^^^^^^^^^^^^^^^^^^^^^
                                                        这就是集群 ID
```

### 2. 更新配置文件

编辑 `~/.databrickscfg`，添加 `cluster_id`：

```ini
[dbr_connect]
host=https://adb-622251785000174.2.databricks.azure.cn/
token=YOUR_DATABRICKS_TOKEN_HERE
cluster_id=YOUR_GPU_CLUSTER_ID_HERE
```

### 3. 验证配置

运行测试脚本：
```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export PATH="$JAVA_HOME/bin:$PATH"
python simple_gpu_test.py
```

如果成功，您应该看到类似输出：
```json
{
  "torch_available": true,
  "cuda_available": true,
  "gpu_count": 1
}
```

## 永久设置 JAVA_HOME

为了避免每次都设置环境变量，添加到您的 shell 配置：

```bash
# 对于 zsh (macOS 默认)
echo 'export JAVA_HOME=/opt/homebrew/opt/openjdk@17' >> ~/.zshrc
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# 验证
java -version
```

## 常见节点类型（用于识别 GPU 集群）

| 节点类型 | 说明 |
|----------|------|
| Standard_* | CPU 集群 |
| GPU_* | GPU 集群 |
| p3.2xlarge | AWS GPU |
| Standard_NC_v3 | Azure GPU |
| Standard_L4s | Google GPU |
