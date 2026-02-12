# DynamicVAE 代码深度分析

> **代码级详细解析** | 整合自多个文档
>
> **目标文件**: `DyAD/model/dynamic_vae.py`
>
> **最后更新**: 2025-02-12

---

## 目录

- [1. 类结构与初始化](#1-类结构与初始化)
- [2. 前向传播详解](#2-前向传播详解)
- [3. 训练与推理模式](#3-训练与推理模式)
- [4. 关键代码片段](#4-关键代码片段)

---

## 1. 类结构与初始化

### 1.1 类定义

```python
class DynamicVAE(nn.Module):
    """
    Dynamic Variational Autoencoder for Battery Fault Detection

    架构: Encoder → Variational Latent Space → Decoder
    输入: 时序数据 [batch, seq_len, features]
    输出: 重构数据 [batch, seq_len, output_features]
    """
```

### 1.2 初始化参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `rnn_type` | str | 必需 | RNN 类型: 'lstm', 'gru', 'rnn' |
| `hidden_size` | int | 必需 | RNN 隐藏状态维度 |
| `latent_size` | int | 必需 | 潜在变量 z 的维度 |
| `encoder_embedding_size` | int | 必需 | 编码器输入特征数量 |
| `decoder_embedding_size` | int | 必需 | 解码器输入特征数量 |
| `output_embedding_size` | int | 必需 | 模型输出特征数量 |
| `num_layers` | int | 1 | RNN 堆叠层数 |
| `bidirectional` | bool | False | 是否使用双向 RNN |
| `variable_length` | bool | False | 是否处理变长序列 |

### 1.3 关键组件初始化

```python
# 动态选择 RNN 类型
rnn = eval('nn.' + rnn_type.upper())

# 编码器 RNN
self.encoder_rnn = rnn(
    encoder_embedding_size,
    hidden_size,
    num_layers=num_layers,
    bidirectional=bidirectional,
    batch_first=True
)

# 计算隐藏层因子
self.hidden_factor = (2 if bidirectional else 1) * num_layers

# 变分相关层
self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
self.hidden2log_v = nn.Linear(hidden_size * self.hidden_factor, latent_size)
self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)

# 输出投影层
self.outputs2embedding = nn.Linear(
    hidden_size * (2 if bidirectional else 1),
    output_embedding_size
)

# 辅助预测器
self.mean2latent = nn.Sequential(
    nn.Linear(latent_size, int(hidden_size / 2)),
    nn.ReLU(),
    nn.Linear(int(hidden_size / 2), 1)
)
```

---

## 2. 前向传播详解

### 2.1 forward 方法签名

```python
def forward(self, input_sequence, encoder_filter, decoder_filter,
            seq_lengths=None, noise_scale=1.0)
```

### 2.2 编码阶段

```python
# 1. 特征筛选
en_input = encoder_filter(input_sequence)  # [B, T, F] → [B, T, 7]

# 2. 变长序列处理（可选）
if self.variable_length and seq_lengths is not None:
    en_input = pack_padded_sequence(en_input, seq_lengths,
                                      batch_first=True,
                                      enforce_sorted=False)

# 3. RNN 编码
output, hidden = self.encoder_rnn(en_input)
# output: [B, T, H*factor]
# hidden: [layers*directions, B, H]

# 4. 解包变长序列（如果需要）
if self.variable_length and seq_lengths is not None:
    output, _ = pad_packed_sequence(output, batch_first=True)

# 5. 提取最终隐藏状态
hidden = hidden.view(hidden.size(1), hidden.size(2))  # Reshape
# hidden: [B, H*factor] for distribution mapping
```

### 2.3 变分推断阶段

```python
# 1. 计算分布参数
mean = self.hidden2mean(hidden)      # [B, H*factor] → [B, Z]
log_v = self.hidden2log_v(hidden)    # [B, H*factor] → [B, Z]

# 2. 重参数化采样
if self.training:
    # 训练模式: 添加噪声
    batch_size = mean.size(0)
    z_noise = to_var(torch.randn([batch_size, self.latent_size]))
    std = torch.exp(0.5 * log_v)
    z = z_noise * std * noise_scale + mean
    # 公式: z = μ + σ × ε, where ε ~ N(0,1)
else:
    # 推理模式: 使用均值
    z = mean
```

### 2.4 解码阶段

```python
# 1. 初始化解码器隐藏状态
decoder_hidden = self.latent2hidden(z)
# [B, Z] → [B, H*factor*layers]
decoder_hidden = decoder_hidden.view(
    self.num_layers * (2 if self.bidirectional else 1),
    -1, z.size(0)
).unsqueeze(0).expand_as(hidden)

# 2. 准备解码器输入
de_input = decoder_filter(input_sequence)  # 选择条件特征
de_input_embedding = self.Decoder_embedding(de_input)

# 3. RNN 解码
outputs, _ = self.decoder_rnn(de_input_embedding, decoder_hidden)
# outputs: [B, T, H*2] (if bidirectional)

# 4. 输出投影
log_p = self.outputs2embedding(outputs)
# [B, T, H*2] → [B, T, output_features]
```

### 2.5 辅助预测

```python
mean_pred = self.mean2latent(mean)
# [B, Z] → [B, 1] (里程/其他标签预测)
```

---

## 3. 训练与推理模式

### 3.1 模式对比

| 方面 | 训练模式 | 推理模式 |
|------|----------|----------|
| **z 计算** | $z = \mu + \sigma \cdot \varepsilon$ | $z = \mu$ |
| **随机性** | 引入噪声探索潜空间 | 确定性输出 |
| **目的** | 学习完整分布 | 使用最可能的值 |

### 3.2 noise_scale 参数

```python
# noise_scale 控制训练时噪声强度
z = z_noise * std * noise_scale + mean

# 值越大 → 噪声越强 → 探索范围越广
# 值越小 → 噪声越弱 → 训练越稳定
```

---

## 4. 关键代码片段

### 4.1 维度变化追踪

假设 `batch_size=32, seq_len=128, hidden_size=128, latent_size=8, bidirectional=True`:

```python
# 输入
input_sequence: [32, 128, 7]

# 编码器输出
encoder_rnn output: [32, 128, 256]  # hidden*2
hidden (reshaped): [32, 256]

# 分布参数
mean: [32, 8]
log_v: [32, 8]

# 采样
z: [32, 8]

# 解码器
decoder_hidden: [32, 256]
decoder_rnn output: [32, 128, 256]  # if bidirectional
log_p (output): [32, 128, 5]  # output_features=5
```

### 4.2 特征选择示例

```python
# BatterybrandaTask 特征选择
def encoder_filter(x):
    """编码器使用全部7个特征"""
    return x  # [B, T, 7]

def decoder_filter(x):
    """解码器使用前2个特征作为条件"""
    return x[:, :, :2]  # [B, T, 2]

def target_filter(x):
    """重构目标是后5个特征"""
    return x[:, :, 2:]  # [B, T, 5]
```

### 4.3 损失计算相关输出

```python
# forward 返回值
return log_p,      # 重构输出 [B, T, O]
       mean,      # 潜在均值 [B, Z]
       log_v,      # 潜在方差 [B, Z]
       z,          # 采样潜在变量 [B, Z]
       mean_pred   # 辅助预测 [B, 1]
```

---

## 代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| **模型定义** | `DyAD/model/dynamic_vae.py` | 34-130 |
| **初始化** | `DyAD/model/dynamic_vae.py` | 34-108 |
| **前向传播** | `DyAD/model/dynamic_vae.py` | 110-130 |
| **特征选择** | `DyAD/model/tasks.py` | 55-67 |
| **数据加载** | `DyAD/model/dataset.py` | 13-42 |

---

**文档版本**: v1.0
**整合来源**: DyAD_CODE_ANALYSIS.md, DyAD_Analysis.md, 01_模型架构.md
