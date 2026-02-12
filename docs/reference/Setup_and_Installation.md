# 环境配置与安装指南

> **单一来源的安装配置参考** | 整合自多个文档
>
> **最后更新**: 2025-02-12

---

## 目录

- [1. 环境依赖](#1-环境依赖)
- [2. 数据准备](#2-数据准备)
- [3. 安装步骤](#3-安装步骤)
- [4. 常见问题解决](#4-常见问题解决)
- [5. CPU轻量配置](#5-cpu轻量配置)

---

## 1. 环境依赖

### 1.1 硬件要求

| 资源 | 最低配置 | 推荐配置 | 说明 |
|------|----------|----------|------|
| **GPU** | GTX 1060 (6GB) | RTX 2060 (6GB) 或更高 | CPU训练也可运行，但慢10x+ |
| **内存** | 8GB | 16GB | 数据加载时需要2-4GB |
| **存储** | 10GB | 20GB | 数据集+模型+输出 |

### 1.2 软件依赖

| 组件 | 版本要求 | 检查方法 | 说明 |
|------|----------|----------|------|
| **Python** | 3.6+ | `python --version` | 推荐 3.8-3.9 |
| **CUDA** | 10.2+ (可选) | `nvcc --version` | PyTorch-Geometric 编译依赖 |
| **PyTorch** | 1.5+ | `python -c "import torch; print(torch.__version__)"` | 需支持 `pack_padded_sequence` |
| **cuDNN** | 7.x+ | - | CUDA 配套版本 |

### 1.3 Python 包依赖

```bash
# 基础依赖
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install tqdm matplotlib scipy
pip install pyod  # 用于 IForest 异常检测
```

---

## 2. 数据准备

### 2.1 下载数据集

**下载链接**：
- OneDrive: https://1drv.ms/u/s!AiSrJIRVqlQAgcjKGKV0fZmw5ifDd8Y?e=CnzELH237
- 北大网盘: https://disk.pku.edu.cn:443/link/37D733DF405D8D7998B8F57E4487515A238

### 2.2 目录结构

解压后应得到以下结构：

```text
data/
├── battery_brand1/
│   ├── train/           # 训练数据（PKL文件）
│   ├── test/            # 测试数据
│   ├── label/           # 标签文件
│   └── column.pkl      # 特征列名称
├── battery_brand2/
│   └── ...
└── battery_brand3/
    └── ...
```

### 2.3 数据格式

每个 PKL 文件应包含：

```python
(
    tensor([[soc, current, min_temp, ...],  # 形状: [T, F]
            [soc, current, min_temp, ...],
            ...]),  # T = 时间步数, F = 特征维度
    {
        'label': [0, 0, 1, ...],  # 长度 T 的标签列表 (0=正常, 1=异常)
        'car': 车辆ID (int),
        'mileage': 里程数 (float),
        'timestamp': 时间戳列表,
        'charge_segment': 充电段ID (可选)
    }
)
```

---

## 3. 安装步骤

### 3.1 一键安装（推荐）

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 2. 安装 PyTorch（根据 CUDA 版本选择）
pip install torch==1.8.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html

# 3. 安装其他依赖
pip install -r requirement.txt
```

### 3.2 PyTorch-Geometric 安装

> ⚠️ **关键**：必须先安装 PyTorch，再安装 PyG 组件

```bash
# CUDA 10.2 版本
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install torch-geometric==1.5.0
```

### 3.3 五折数据划分

```bash
cd data
jupyter notebook five_fold_train_test_split.ipynb
# 运行所有单元格，等待几分钟
```

**生成文件位置**：
```text
five_fold_utils/
├── all_car_dict.npz.npy
├── ind_odd_dict1.npz.npy
├── ind_odd_dict2.npz.npy
└── ind_odd_dict3.npz.npy
```

---

## 4. 常见问题解决

### 4.1 CUDA 版本不匹配

**现象**：
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**原因**：PyTorch 编译的 CUDA 版本与系统 CUDA 不一致

**解决**：
```bash
# 卸载当前 PyTorch
pip uninstall torch torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# 重新安装匹配版本
# 如果系统 CUDA 是 11.x
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

### 4.2 内存溢出 (OOM)

**现象**：
```
RuntimeError: CUDA out of memory
```

**解决**：修改 `model_params_battery_brand*.json`:
```json
{
    "batch_size": 64,   // 减小批次
    "jobs": 4           // 减少进程数
}
```

### 4.3 路径依赖问题

**现象**：
```
FileNotFoundError: [Errno 2] No such file or directory: '../data/battery_brand1/train'
```

**原因**：必须在 `DyAD/` 目录下运行

**解决**：
```bash
cd DyAD  # 确保在此目录
python main_five_fold.py ...
```

### 4.4 DataLoader 卡住

**现象**：训练时进度条不动

**解决**：
```python
# 临时测试：设置 jobs=0 (主进程加载)
DataLoader(dataset=train, batch_size=128, num_workers=0, ...)

# 或减少进程数
"jobs": 4  # 在 config.json 中
```

---

## 5. CPU 轻量配置

### 5.1 禁用 CUDA

```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=""

# 或者在 Python 脚本开头添加
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

### 5.2 轻量级参数配置

修改 `model_params_battery_brand*.json`:

```json
{
    "batch_size": 16,        // 减小批次
    "hidden_size": 32,        // 减小隐藏层
    "latent_size": 4,         // 减小潜在维度
    "num_layers": 1,          // 减少层数
    "epochs": 1,              // 单轮测试
    "jobs": 0                // CPU环境设为0
}
```

### 5.3 数据截取（可选）

在 `dataset.py` 中限制加载数据量：

```python
# 仅加载前 10 个样本用于测试
self.battery_dataset = self.battery_dataset[:10]
```

---

## 运行前检查清单

- [ ] 确认 CUDA 版本已安装（`nvcc --version`）
- [ ] 确认 GPU 可用（`nvidia-smi`）
- [ ] 确认 PyTorch 可用 CUDA（`python -c "import torch; print(torch.cuda.is_available())"`）
- [ ] 确认数据集已下载并解压到 `data/`
- [ ] 确认 `five_fold_utils/*.npz.npy` 已生成
- [ ] 确认要使用的品牌已在配置文件中设置

---

**文档版本**: v1.0
**整合来源**: Engineering_Overview.md, codebase_analysis.md, DyAD_Five_Fold_Training_Evaluation.md
