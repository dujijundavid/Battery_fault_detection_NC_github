# Battery Fault Detection - 快速开始

> **30分钟上手指南** | 面向工程师和开发者

---

## 项目概述

本项目实现了基于 **DyAD (Dynamic Variational Autoencoder)** 的电池故障检测系统，使用五折交叉验证评估模型在充电数据上的异常检测能力。

**核心特点**：
- 🔄 **双向RNN编码** + **条件解码**
- 🎯 **五折交叉验证** 确保鲁棒性
- 📊 **重构误差评分** 无需阈值训练
- 🏭 **多品牌支持** 品牌1/2/3独立模型

---

## 一分钟检查清单

### 环境检查

```bash
# 检查 Python 版本 (需要 3.6+)
python --version

# 检查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 检查数据目录
ls data/battery_brand1/train
ls five_fold_utils/ind_odd_dict1.npz.npy
```

### 快速修复

| 问题 | 解决方案 |
|------|----------|
| PyTorch 未安装 | `pip install torch` |
| CUDA 不可用 | 设置环境变量 `export CUDA_VISIBLE_DEVICES=""` 使用 CPU |
| 数据缺失 | 参考 [Setup_and_Installation.md](reference/Setup_and_Installation.md) 下载数据 |

---

## 5分钟快速运行

### 1. 进入项目目录

```bash
cd /path/to/Battery_fault_detection_NC_github/DyAD
```

### 2. 运行单折训练

```bash
python main_five_fold.py \
    --config_path model_params_battery_brand1.json \
    --fold_num 0
```

### 3. 查看输出

训练完成后，检查输出目录：

```bash
ls dyad_vae_save/*_fold_0/model/model.torch
ls dyad_vae_save/*_fold_0/result/test_segment_scores.csv
```

---

## 核心文件速查

### 必知文件

| 文件 | 作用 | 修改频率 |
|------|------|----------|
| `main_five_fold.py` | 训练入口 | ⭐ 必须运行 |
| `model/dynamic_vae.py` | 模型定义 | 架构修改时 |
| `train.py` | 训练逻辑 | 调试时 |
| `model_params_battery_brand*.json` | 超参数配置 | 🔧 调参时 |

### 配置文件示例

```json
{
    "latent_size": 8,
    "hidden_size": 128,
    "batch_size": 128,
    "epochs": 3,
    "learning_rate": 0.005,
    "nll_weight": 10
}
```

---

## 常用命令

### 单品牌五折训练

```bash
# 品牌1
for fold in {0..4}; do
    python main_five_fold.py \
        --config_path model_params_battery_brand1.json \
        --fold_num $fold
done
```

### CPU 轻量模式

```bash
# 修改配置文件中的 batch_size=16, hidden_size=32
# 然后运行
python main_five_fold.py \
    --config_path model_params_battery_brand1.json \
    --fold_num 0
```

---

## 输出解读

### 训练成功标志

- ✅ `dyad_vae_save/YYYY-MM-DD-HH-MM-SS_fold0/` 目录创建
- ✅ `model/model.torch` 文件存在
- ✅ `result/test_segment_scores.csv` 包含评分结果

### 评分文件格式

```csv
car,label,rec_error
123,0,0.0234
456,1,0.1456
...
```

- `rec_error` 越大 → 越可能是异常
- 后续可计算 AUROC 评估性能

---

## 下一步

### 学习路径

- **初学者**: 从 [tutorials/00_基础概念.md](tutorials/00_基础概念.md) 开始
- **开发者**: 阅读 [reference/](reference/) 目录下的参考文档
- **研究人员**: 查看 [technical/](technical/) 目录的深度分析

### 调参优化

详见 [Training_and_Evaluation.md](reference/Training_and_Evaluation.md#5-超参数配置)

### 问题排查

详见 [Setup_and_Installation.md](reference/Setup_and_Installation.md#4-常见问题解决)

---

## 获取帮助

| 资源 | 链接 |
|------|------|
| 完整文档 | [INDEX.md](INDEX.md) |
| 架构参考 | [reference/Architecture_Reference.md](reference/Architecture_Reference.md) |
| 训练评估 | [reference/Training_and_Evaluation.md](reference/Training_and_Evaluation.md) |
| 环境配置 | [reference/Setup_and_Installation.md](reference/Setup_and_Installation.md) |

---

**文档版本**: v1.0
**最后更新**: 2025-02-12
