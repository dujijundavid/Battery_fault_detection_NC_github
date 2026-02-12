# 数学推导与原理

> **DyAD 模型数学基础** | 整合自多个文档
>
> **最后更新**: 2025-02-12

---

## 目录

- [1. VAE 基础](#1-vae-基础)
- [2. 损失函数推导](#2-损失函数推导)
- [3. KL 散度](#3-kl-散度)
- [4. 重参数化技巧](#4-重参数化技巧)
- [5. 退火机制](#5-退火机制)

---

## 1. VAE 基础

### 1.1 变分自编码器目标

VAE 的核心思想是学习数据的潜在分布 $q(z|x)$，使其接近真实的后验分布 $p(z|x)$。

**Evidence Lower Bound (ELBO)**:

$$
\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))
$$

### 1.2 DyAD 中的变分推断

DyAD 假设潜在变量 $z$ 服从标准正态分布：

$$
p(z) = \mathcal{N}(0, I)
$$

编码器学习变分后验分布：

$$
q(z|x) = \mathcal{N}(z; \mu(x), \text{diag}(\sigma^2(x)))
$$

其中：
- $\mu(x)$ = `hidden2mean(hidden)` → 均值向量
- $\sigma^2(x)$ = `exp(hidden2log_v(hidden))` → 方差向量

---

## 2. 损失函数推导

### 2.1 完整损失函数

DyAD 使用三项损失：

$$
\mathcal{L} = \mathcal{L}_{\text{NLL}} + \beta \cdot \mathcal{L}_{\text{KL}} + \mathcal{L}_{\text{Label}}
$$

### 2.2 重构损失 (NLL)

使用均方误差损失：

$$
\mathcal{L}_{\text{NLL}} = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2
$$

其中：
- $\mathbf{x}_i$ = 原始输入序列
- $\hat{\mathbf{x}}_i$ = 解码器重构输出
- $N$ = 批次大小 × 序列长度

**代码实现** (使用 SmoothL1Loss 更鲁棒):

```python
nll_loss = torch.nn.SmoothL1Loss(reduction='mean')(log_p, target)
```

### 2.3 标签损失 (辅助任务)

$$
\mathcal{L}_{\text{Label}} = \|\text{mean\_pred} - \text{mileage}\|^2
$$

用于预测连续标签（如里程数），约束潜在空间学习有意义表示。

---

## 3. KL 散度

### 3.1 公式推导

两个正态分布之间的 KL 散度有解析解：

$$
D_{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2}
$$

### 3.2 DyAD 中的应用

DyAD 中，先验分布 $p(z) = \mathcal{N}(0, 1)$，后验分布 $q(z|x) = \mathcal{N}(\mu, \sigma^2)$，代入得：

$$
\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{Z} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)
$$

**代码实现**:

```python
kl_loss = -0.5 * torch.sum(1 + log_v - mean.pow(2) - log_v.exp())
```

### 3.3 KL 散度的作用

| 训练阶段 | KL 权重 | 效果 |
|----------|----------|------|
| **初期** | 低 (0.01) | 允许潜在变量偏离先验，专注重构 |
| **后期** | 高 (1.0) | 强制潜在分布接近标准正态，正则化 |

---

## 4. 重参数化技巧

### 4.1 问题陈述

采样操作 $z \sim q(z|x)$ 不可微，无法直接反向传播。

### 4.2 重参数化技巧

将随机性分离：

$$
z = \mu + \sigma \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

其中：
- $\mu, \sigma$ 是确定性的网络输出
- $\varepsilon$ 是独立采样的噪声
- $\odot$ 是逐元素乘法

### 4.3 梯度传播

现在梯度可以流向 $\mu$ 和 $\sigma$：

$$
\frac{\partial \mathcal{L}}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial z}
$$

$$
\frac{\partial \mathcal{L}}{\partial \sigma} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial \sigma} = \frac{\partial \mathcal{L}}{\partial z} \cdot \varepsilon
$$

**代码实现**:

```python
# 训练时
z_noise = torch.randn([batch_size, latent_size])
std = torch.exp(0.5 * log_v)
z = z_noise * std * noise_scale + mean

# 推理时
z = mean  # 等价于无限次采样的期望
```

---

## 5. 退火机制

### 5.1 为什么需要退火

训练初期，如果 KL 损失权重过大：
- 后验分布 $q(z|x)$ 过早收敛到先验 $p(z) = \mathcal{N}(0, 1)$
- 潜在变量 $z$ 失去区分能力
- 重构能力下降

### 5.2 线性退火

$$
\beta(t) = \text{anneal}_0 \cdot \min\left(1, \frac{t}{x_0}\right)
$$

**参数**:
- `anneal0` = 0.01 (初始权重)
- `x0` = 500 (退火中点步数)

**代码**:

```python
kl_weight = anneal0 * min(1.0, step / x0)
```

### 5.3 Logistic 退火

$$
\beta(t) = \frac{\text{anneal}_0}{1 + \exp(-k(t - x_0))}
$$

**参数**:
- `k` = 0.0025 (斜率，控制增长速度)
- `x0` = 500 (中心点，第 x0 步达到一半权重)

**代码**:

```python
kl_weight = anneal0 / (1 + torch.exp(-k * (step - x0)))
```

### 5.4 退火曲线对比

| 步数 | 线性退火 | Logistic 退火 |
|------|----------|---------------|
| 0 | 0.000 | 0.000 |
| 250 | 0.005 | 0.003 |
| 500 | 0.010 | 0.005 |
| 750 | 0.010 | 0.008 |
| 1000+ | 0.010 | 0.010 |

---

## 公式速查表

### 关键公式

| 符号 | 含义 | 代码位置 |
|------|------|----------|
| $z$ | 潜在变量 | `z` |
| $\mu$ | 潜在均值 | `mean` |
| $\sigma^2$ | 潜在方差 | `exp(log_v)` |
| $\epsilon$ | 标准正态噪声 | `z_noise` |
| $\hat{x}$ | 重构输出 | `log_p` |
| $\beta$ | KL 退火权重 | `kl_weight` |

### 维度关系

$$
[B, T, F] \xrightarrow{\text{Encoder}} [B, H] \xrightarrow{\text{VAE}} [B, Z] \xrightarrow{\text{Decoder}} [B, T, O]
$$

- $B$ = batch_size
- $T$ = seq_len
- $F$ = encoder_features (7)
- $H$ = hidden_size
- $Z$ = latent_size
- $O$ = output_features (5)

---

**文档版本**: v1.0
**整合来源**: DyAD_Analysis.md, DyAD_CODE_ANALYSIS.md, 05_数学原理.md, DyAD_Visualization_Roadmap.md
