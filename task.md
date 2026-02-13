# DyAD 电池故障检测研究复现计划

> **项目**: DyAD (Dynamic Variational Autoencoder) - Nature Communications 论文复现
> **时间目标**: 3天（使用 Claude Code 加速）
> **用户背景**: AI工程师，前数据科学家+全栈工程师

---

## 上下文

### 本地环境
- **操作系统**: macOS with Apple M1
- **Python**: 3.12.12（需要降级到 3.8-3.10）
- **GPU**: 无 NVIDIA CUDA 支持
- **训练策略**: 本地开发 + Databricks GPU 集群训练

### 数据来源
- OneDrive: https://1drv.ms/u/s!AiSrJIRVqlQAgcjKGKV0fZmw5ifDd8Y?e=CnzELH237
- 北大网盘: https://disk.pku.edu.cn:443/link/37D733DF405D8D7998B8F57E4487515A238

### 已识别问题
1. **硬编码路径**: [dyad/model/dataset.py:14](dyad/model/dataset.py#L14) 默认品牌1
2. **配置分散**: 品牌切换需修改多处代码
3. **数据缺失**: data/battery_brand*/ 目录不存在

---

## 阶段 1: 本地环境搭建 (Day 1, 2-3小时)

### 1.1 创建 Conda 环境

```bash
# 创建专用环境 (Python 3.8 兼容性最佳)
conda create -n dyad python=3.8 -y
conda activate dyad

# 安装 PyTorch CPU 版本 (M1 兼容)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
cd /Users/David/Desktop/github_repos/Battery_fault_detection_NC_github
pip install -r requirements_relaxed.txt
```

### 1.2 下载数据集

```bash
cd /Users/David/Desktop/github_repos/Battery_fault_detection_NC_github/data

# 从 OneDrive 或北大网盘下载并解压
# 验证结构: battery_brand1/, battery_brand2/, battery_brand3/
```

### 1.3 生成五折划分文件

```bash
cd /Users/David/Desktop/github_repos/Battery_fault_detection_NC_github/data

# 运行 notebook 生成分割文件
jupyter notebook five_fold_train_test_split.ipynb
# 执行所有单元格

# 验证输出
ls five_fold_utils/
# 应包含: all_car_dict.npz.npy, ind_odd_dict{1,2,3}.npz.npy
```

---

## 阶段 2: Databricks GPU 环境 (Day 1, 1-2小时)

### 2.1 配置 Databricks Connect

编辑 `~/.databrickscfg`:

```ini
[dbr_connect]
host = https://adb-622251785000174.2.databricks.azure.cn
token = YOUR_TOKEN
cluster_id = YOUR_GPU_CLUSTER_ID
```

### 2.2 验证 GPU 可用性

在 Databricks Notebook 界面运行 [infrastructure/databricks/GPU_test.ipynb](infrastructure/databricks/GPU_test.ipynb)

### 2.3 代码同步策略

```bash
# 方法 1: 使用 Databricks Bundle
databricks bundle deploy

# 方法 2: 在 Notebook 中克隆
%sh
git clone https://github.com/your-fork/Battery_fault_detection_NC_github.git
```

---

## 阶段 3: 代码质量改进 (Day 2, 2-3小时)

### 3.1 创建统一配置类

**新建文件**: [dyad/config.py](dyad/config.py)

```python
from dataclasses import dataclass
import os
import json

@dataclass
class DyADConfig:
    """DyAD 统一配置管理"""

    # 路径配置
    base_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_base_path: str = os.path.join(base_path, "../data")
    brand: int = 1

    @property
    def split_file(self) -> str:
        if self.brand == 'all':
            return os.path.join(self.data_base_path, "splits/all_car_dict.npz.npy")
        return os.path.join(self.data_base_path, f"splits/ind_odd_dict{self.brand}.npz.npy")

    @classmethod
    def from_json(cls, path: str, brand: int) -> 'DyADConfig':
        with open(path) as f:
            data = json.load(f)
        return cls(brand=brand, **data)
```

### 3.2 重构数据加载器

**修改文件**: [dyad/model/dataset.py](dyad/model/dataset.py)

```python
# 修改前
def __init__(self, data_path, all_car_dict_path='../five_fold_utils/all_car_dict.npz.npy',
     ind_ood_car_dict_path='../five_fold_utils/ind_odd_dict1.npz.npy', ...):

# 修改后
def __init__(self, config: DyADConfig, train=True, fold_num=0):
    ind_ood_car_dict_path = config.split_file
    # ...
```

### 3.3 更新入口文件

**修改文件**: [dyad/main_five_fold.py](dyad/main_five_fold.py)

```python
# 添加 --brand 参数
argparse.add_argument('--brand', type=int, default=1, help='Battery brand (1, 2, 3, or all)')

# 替代 --config_path
config = DyADConfig.from_json(f"model_params_battery_brand{args.brand}.json", args.brand)
```

---

## 阶段 4: 实验复现 (Day 2-3, 3-6小时)

### 4.1 本地快速验证

```bash
cd /Users/David/Desktop/github_repos/Battery_fault_detection_NC_github/dyad

# CPU 测试 (轻量级配置)
export CUDA_VISIBLE_DEVICES=""
python main_five_fold.py --brand 1 --fold_num 0 --epochs 1
```

### 4.2 Databricks GPU 五折训练

在 Databricks Notebook 中:

```python
# 五折交叉验证
for fold in range(5):
    print(f"=== Fold {fold} ====")
    !python main_five_fold.py --brand 1 --fold_num {fold}
```

### 4.3 评估 AUROC

更新 [evaluation/dyad_eval_fivefold-threshold.ipynb](evaluation/dyad_eval_fivefold-threshold.ipynb) 中的路径并运行

**预期指标**:
- 品牌1 AUROC > 0.85
- 品牌2 AUROC > 0.80
- 品牌3 AUROC > 0.75

---

## 阶段 5: 文档和扩展 (Day 3, 1-2小时)

### 5.1 添加代码文档

为以下文件添加 Google-style docstring:
- [dyad/model/dynamic_vae.py](dyad/model/dynamic_vae.py) - DynamicVAE 模型
- [dyad/train.py](dyad/train.py) - 训练逻辑
- [dyad/extract.py](dyad/extract.py) - 特征提取

### 5.2 新异常检测方法扩展点

**新建文件**: [dyad/model/base_anomaly_detector.py](dyad/model/base_anomaly_detector.py)

```python
from abc import ABC, abstractmethod

class BaseAnomalyDetector(ABC):
    """异常检测方法抽象基类"""

    @abstractmethod
    def train(self, train_data) -> None: pass

    @abstractmethod
    def compute_anomaly_score(self, test_data): pass
```

---

## 关键文件清单

| 文件 | 作用 | 修改类型 |
|------|------|----------|
| [dyad/model/dataset.py](dyad/model/dataset.py) | 数据加载 | 重构 |
| [dyad/main_five_fold.py](dyad/main_five_fold.py) | 训练入口 | 重构 |
| [dyad/config.py](dyad/config.py) | 统一配置 | 新建 |
| [dyad/model/dynamic_vae.py](dyad/model/dynamic_vae.py) | 模型定义 | 文档化 |
| [data/five_fold_train_test_split.ipynb](data/five_fold_train_test_split.ipynb) | 数据预处理 | 执行 |

---

## 时间估算 (使用 Claude Code 加速)

| 阶段 | 常规时间 | 加速后 |
|--------|-----------|---------|
| 1. 本地环境 | 4-6h | 2-3h |
| 2. Databricks GPU | 2-3h | 1-2h |
| 3. 代码质量改进 | 4-6h | 2-3h |
| 4. 实验复现 | 6-10h | 3-5h |
| 5. 文档扩展 | 3-4h | 1-2h |
| **总计** | **19-29h** | **9-15h** |

---

## 验证清单

完成每个阶段后验证：

- [ ] **阶段1**: Conda 环境可导入 torch, tensorflow
- [ ] **阶段1**: five_fold_utils/ 包含 .npz.npy 文件
- [ ] **阶段2**: Databricks GPU_test.ipynb 显示 CUDA 可用
- [ ] **阶段3**: `--brand 2` 可正确切换配置
- [ ] **阶段4**: test_segment_scores.csv 包含所有测试样本
- [ ] **阶段4**: AUROC 分数达到预期范围
- [ ] **阶段5**: 代码模块有完整的 docstring

---

## 下一步

计划获批后，按顺序执行：
1. 执行阶段1（环境搭建）
2. 完成后进入阶段2（Databricks）
3. 根据进度调整后续阶段

---

## 参考资料

- Nature 论文: https://www.nature.com/articles/s41598-024-82960-0
- 原始仓库: https://github.com/962086838/Battery_fault_detection_NC_github
- CLAUDE.md: [CLAUDE.md](CLAUDE.md)
- 架构参考: [docs/reference/Architecture_Reference.md](docs/reference/Architecture_Reference.md)
