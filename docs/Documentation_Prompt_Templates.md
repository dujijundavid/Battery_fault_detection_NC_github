# 文档撰写通用 Prompt 模板

> **文档来源**：基于 [DyAD_Visualization_Roadmap.md](DyAD_Visualization_Roadmap.md) 高质量文档提炼  
> **适用场景**：技术文档、教程、代码分析、架构说明等  
> **最后更新**：2025-11-24

---

## 📋 目录

- [核心设计原则](#核心设计原则)
- [Prompt 模板版本](#prompt-模板版本)
  - [版本 1：基础教程版](#版本-1基础教程版)
  - [版本 2：进阶技术分析版](#版本-2进阶技术分析版)
  - [版本 3：专家级深度解析版](#版本-3专家级深度解析版)
  - [版本 4：可视化优先版](#版本-4可视化优先版)
  - [版本 5：代码实践版](#版本-5代码实践版)
- [复用组件库](#复用组件库)
- [质量检查清单](#质量检查清单)

---

## 核心设计原则

基于 `DyAD_Visualization_Roadmap.md` 提炼的核心成功要素：

### 1. **螺旋式学习结构** 🌀

同一概念在不同层次反复出现，逐步加深：
- **第一圈**：是什么（整体认知）
- **第二圈**：为什么（原理深入）
- **第三圈**：怎么做（实践验证）

### 2. **多模态表达** 🎨

- **Mermaid 图表**：架构图、流程图、序列图、决策树
- **数学公式**：LaTeX 格式，配合文字解释
- **代码示例**：完整可运行的代码块，标注关键行
- **表格对比**：功能对比、参数说明、问题诊断

### 3. **渐进式复杂度** 📈

- 从高层概览到底层细节
- 从简化配置到完整参数
- 从单一示例到多场景对比

### 4. **检查点机制** ✅

每个主要章节结束设置检查点：
- 关键概念清单
- 动手验证任务
- 疑问引导提示

### 5. **问题驱动导航** 🧭

- 决策树式问题诊断
- 常见错误与解决方案
- 延伸阅读路径

---

## Prompt 模板版本

### 版本 1：基础教程版

**适用场景**：为初学者编写入门教程、快速上手指南

```
请为 [技术主题/工具/框架] 创建一份渐进式教程文档，要求：

## 文档定位
- 目标读者：[初学者/中级开发者/...]
- 预计学习时间：[X-Y 小时]
- 前置知识：[列出必要的背景知识]

## 内容结构（螺旋式三层）

### 第一圈：建立整体认知
1. **解决什么问题**
   - 业务场景描述
   - 输入输出示例
   - 核心挑战（用 ✗ 标记痛点）
   
2. **技术方案核心思想**
   - 用引用块突出核心理念
   - 与替代方案对比（表格形式）

3. **整体架构**
   - 高层数据流图（Mermaid）
   - 核心组件说明表格（输入/输出/作用）

4. **快速示例**
   - 最小化可运行代码
   - 预期输出说明

5. **检查点**（用 checkbox 列表）

### 第二圈：深入原理
1. **核心概念详解**
   - 数学/理论基础（LaTeX 公式 + 直观解释）
   - 关键算法步骤（序列图或流程图）

2. **设计决策分析**
   - "为什么这样设计？"
   - 对比"不这样做会怎样"（用对比图表）

3. **参数配置详解**
   - 表格：参数名、典型值、作用、调优方向
   
4. **检查点**

### 第三圈：实践验证
1. **环境准备**
   - 最小化配置（CPU/小数据集测试）
   - 完整配置

2. **逐步实践**
   - 带注释的完整代码
   - 可视化输出示例（matplotlib 代码）

3. **问题诊断**
   - 决策树（Mermaid graph TD）
   - 常见错误表格（现象/原因/解决方案）

4. **检查点**

## 可视化要求
- 每个主要章节至少 1 个 Mermaid 图
- 关键流程用序列图（sequenceDiagram）
- 架构用组件图（graph TB/LR）
- 问题诊断用决策树（graph TD）

## 写作风格
- 使用 emoji 增强可读性（📚 🎨 ✅ ❌ 等）
- 关键术语首次出现用粗体
- 代码块必须指定语言并加注释
- 复杂概念用类比解释

## 输出格式
Markdown，包含：
- 完整目录（带锚点链接）
- 三级标题结构
- 代码块可直接复制运行
```

---

### 版本 2：进阶技术分析版

**适用场景**：深入分析已有代码库、技术方案对比、架构说明

```
请对 [代码库/模块/系统] 进行深入技术分析，生成文档：

## 分析维度

### 1. 架构层
- **系统架构图**（Mermaid）
  - 模块依赖关系
  - 数据流向
  - 关键接口
  
- **设计模式识别**
  - 使用了哪些设计模式
  - 为何选择这些模式
  
### 2. 实现层
- **核心类/函数分析**
  - 表格：类名/职责/输入输出/依赖
  - 关键方法的伪代码
  
- **张量维度追踪**（适用于 ML 项目）
  - 完整的数据变换流程图
  - 每一步的维度标注
  
- **算法复杂度**
  - 时间/空间复杂度分析
  - 瓶颈识别

### 3. 数学/理论层（如适用）
- **公式推导**
  - 完整的 LaTeX 公式
  - 逐步推导过程
  - 物理意义解释
  
- **数值稳定性分析**
  - 潜在问题（溢出/除零）
  - 改进方案代码对比

### 4. 工程层
- **配置参数详解**
  - 参数表（名称/默认值/影响/调优建议）
  
- **错误处理机制**
  - 异常类型与处理策略
  
- **性能优化点**
  - 已有优化（加亮显示）
  - 潜在优化方向

## 可视化要求
- 架构图：分层组件图
- 数据流：带维度标注的流程图
- 算法流程：序列图或活动图
- 决策逻辑：决策树

## 对比分析（如适用）
对比：[当前方案] vs [替代方案]
- 表格对比（功能/性能/复杂度）
- 适用场景分析

## 输出格式
- 目录深度至少 3 层
- 代码文件链接（file:/// 格式，带行号）
- 关键代码段嵌入文档并加注释
- 每个主要部分后附"关键要点"总结框
```

---

### 版本 3：专家级深度解析版

**适用场景**：论文复现、前沿技术解读、算法优化

```
为 [算法/模型/系统] 创建专家级技术文档，要求：

## 深度分析框架

### 理论基础
1. **问题形式化**
   - 数学定义（集合论/概率论符号）
   - 目标函数
   - 约束条件

2. **理论推导**
   - 从第一性原理出发
   - 完整的证明过程
   - 定理/引理标注

3. **与经典方法对比**
   - 理论差异表格
   - 复杂度对比（Big-O 表示）

### 实现细节
1. **算法伪代码**
   - 分层次：高层 → 细节
   - 关键步骤复杂度标注

2. **数值实现技巧**
   - 重参数化技巧
   - 数值稳定性处理（对比不稳定/稳定实现）
   - 梯度计算细节

3. **代码与公式映射**
   - 表格：公式项 → 代码行 → 文件位置

### 实验验证
1. **消融实验**
   - 各组件的贡献分析
   - 可视化对比（matplotlib 代码）

2. **超参数敏感性**
   - 参数扫描曲线
   - 交互效应分析

3. **边界情况测试**
   - 极端输入（零/无穷/NaN）
   - 边界条件验证

### 扩展方向
1. **理论扩展**
   - 放宽假设的影响
   - 更一般的形式

2. **工程扩展**
   - 代码示例（如添加注意力机制）

## 可视化要求
- 理论框架：Mermaid 类图
- 算法流程：详细的活动图（包含决策节点）
- 数值对比：表格 + 折线图代码
- 消融实验：柱状图/热力图代码

## 文献引用
- 关键论文引用（作者，年份，标题）
- 延伸阅读列表（分层次：入门/进阶/前沿）

## 附录
- 完整公式表
- 符号表（Notation Table）
- 实验复现清单（环境/数据/命令/预期结果）
```

---

### 版本 4：可视化优先版

**适用场景**：需要大量图表的文档、教学演示、数据分析报告

```
为 [主题] 创建高度可视化的文档，要求：

## 可视化策略

### 架构/流程可视化
1. **系统级**：graph TB（自顶向下）
   - 使用 subgraph 分组
   - 用颜色区分不同类型（输入/处理/输出）
   - style 语句统一配色

2. **交互级**：sequenceDiagram
   - 清晰的参与者（participant）
   - 激活框（activate/deactivate）
   - 注释框（Note over）说明关键步骤

3. **数据流**：graph LR（从左到右）
   - 维度标注（如：Batch × Time × Features）
   - 关键变换节点高亮

### 决策/逻辑可视化
1. **问题诊断树**
   - 起始节点用菱形（{问题？}）
   - 解决方案节点用矩形，填充绿色
   - 路径标签清晰（--问题类型--\u003e）

2. **算法流程**
   - 条件分支清晰
   - 循环结构明确标注

### 数据可视化（Python 代码）
1. **损失曲线**
   - 多子图布局（最少 1×2）
   - 堆叠面积图 + 分项趋势图
   - 图例/网格/标签完整

2. **分布对比**
   - 直方图 + 箱线图
   - 阈值线标注
   - 统计量文字输出

3. **降维可视化**
   - t-SNE/PCA 散点图
   - 颜色映射（cmap）表示类别
   - colorbar 说明

### 表格可视化
1. **参数配置表**
   - 列：参数名/典型值/作用/调优方向/影响
   - 用 emoji 增强可读性（➕ ✨ ⚠️）

2. **对比表**
   - 行：方案A/方案B
   - 列：评价维度
   - 优势单元格用 ✅，劣势用 ❌

## Mermaid 最佳实践
- 节点命名：语义化（Input/Process/Output）
- 颜色方案：
  - 输入：蓝色系（#e3f2fd）
  - 处理：黄色系（#fff9c4）
  - 输出：绿色系（#c8e6c9）
  - 错误：红色系（#ffebee）
- 文字换行：用 \u003cbr/\u003e 避免节点过宽
- 复杂图：拆分为多个子图

## 代码示例模板
每个可视化代码包含：
1. 数据加载部分（带示例路径）
2. 处理逻辑（清晰注释）
3. 绘图部分（参数完整）
4. 保存与显示（带 dpi=300）
5. 预期输出说明（文字描述图形特征）

## 图表配置清单
- [ ] 所有图表有标题（fontsize=14, fontweight='bold'）
- [ ] 坐标轴标签清晰（fontsize=12）
- [ ] 图例位置合理（不遮挡数据）
- [ ] 网格线适度（alpha=0.3）
- [ ] 颜色对比度足够（考虑色盲友好）
```

---

### 版本 5：代码实践版

**适用场景**：实战教程、API 文档、代码审查报告

```
为 [项目/模块] 创建代码实践文档，要求：

## 实践路径设计

### 快速开始（5 分钟）
1. **最小化复现**
   - CPU 环境配置
   - 小数据集（\u003c=20 样本）
   - 简化参数（JSON 配置文件）
   - 单次运行命令
   - 预期输出示例

2. **代码结构概览**
   - 文件树（用代码块展示）
   - 核心文件简介表格

### 逐步深入（30 分钟）
1. **关键函数解读**
   - 每个函数包含：
     - 签名（输入输出类型标注）
     - 伪代码
     - 实际代码（带逐行注释）
     - 调用示例

2. **数据流追踪**
   - 完整的执行流程（函数调用图）
   - 中间变量打印点建议
   - 调试技巧

### 完整实践（2 小时）
1. **端到端流程**
   - 数据准备
   - 训练脚本
   - 评估脚本
   - 结果可视化

2. **配置调优**
   - 参数扫描表
   - 性能对比（表格）
   - 最佳实践建议

## 代码质量要求
1. **可运行性**
   - 所有代码块独立可运行
   - 依赖明确（import 语句完整）
   - 路径用占位符（如 `xxx_fold0`）

2. **注释策略**
   - 关键行：行末注释（# 说明）
   - 代码块：顶部注释（"""docstring"""）
   - 复杂逻辑：分步注释

3. **错误处理**
   - 常见错误示例（❌ 不稳定实现）
   - 改进方案（✅ 稳定实现）
   - 对比说明

## 问题诊断模块
1. **决策树**（Mermaid）
   - 起点：{遇到什么问题？}
   - 分支：代码报错/训练问题/性能问题
   - 叶节点：具体解决方案（带章节引用）

2. **常见错误表**
   - 列：错误现象/可能原因/解决方案/相关章节

3. **Debug 清单**
   - [ ] 环境检查
   - [ ] 数据检查
   - [ ] 配置检查
   - [ ] 输出检查

## 最佳实践模块
1. **数值稳定性**
   - Clamp 技巧
   - Mask 处理
   - 梯度裁剪
   - 归一化

2. **性能优化**
   - DataLoader 参数
   - 批处理技巧
   - 缓存策略

## 验证模块
1. **单元测试建议**
   - 测试用例列表
   - 边界条件

2. **集成测试**
   - 端到端流程验证
   - 输出一致性检查

## 附录
- 完整配置文件示例
- 环境依赖清单（requirements.txt）
- 常用命令速查表
```

---

## 复用组件库

### Mermaid 图表模板

#### 1. 数据流图（带维度标注）

```mermaid
graph TB
    Input[输入数据\u003cbr/\u003eShape: Batch × Time × Features]
    
    subgraph Processing[处理模块]
    Step1[步骤 1\u003cbr/\u003e操作描述]
    Step2[步骤 2\u003cbr/\u003e操作描述]
    end
    
    Output[输出结果\u003cbr/\u003eShape: Batch × Output_Dim]
    
    Input --\u003e Step1 --\u003e Step2 --\u003e Output
    
    style Input fill:#e3f2fd
    style Output fill:#c8e6c9
```

#### 2. 序列图模板

```mermaid
sequenceDiagram
    participant User as 用户/调用者
    participant System as 系统/模块
    participant DB as 数据库/外部服务
    
    User-\u003e\u003eSystem: 请求（参数）
    activate System
    
    System-\u003e\u003eDB: 查询数据
    activate DB
    DB--\u003e\u003eSystem: 返回结果
    deactivate DB
    
    System-\u003e\u003eSystem: 处理逻辑
    Note over System: 关键步骤说明
    
    System--\u003e\u003eUser: 响应（结果）
    deactivate System
```

#### 3. 决策树模板

```mermaid
graph TD
    Start{问题描述？}
    
    Start --\u003e|情况A| BranchA{子问题A？}
    Start --\u003e|情况B| BranchB{子问题B？}
    
    BranchA --\u003e|条件1| SolutionA1[解决方案A1]
    BranchA --\u003e|条件2| SolutionA2[解决方案A2]
    
    BranchB --\u003e|条件1| SolutionB1[解决方案B1]
    
    style Start fill:#2196f3,color:#fff
    style SolutionA1 fill:#4caf50,color:#fff
    style SolutionA2 fill:#4caf50,color:#fff
    style SolutionB1 fill:#4caf50,color:#fff
```

#### 4. 对比架构图

```mermaid
graph LR
    subgraph Method_A[方法 A]
    A1[组件1] --\u003e A2[组件2] --\u003e A3[输出]
    end
    
    subgraph Method_B[方法 B]
    B1[组件1] --\u003e B2[组件2\u003cbr/\u003e改进点] --\u003e B3[输出]
    end
    
    style Method_A fill:#ffebee
    style Method_B fill:#e8f5e9
    style B2 fill:#fff9c4
```

---

### Python 可视化代码模板

#### 1. 损失曲线图

```python
import matplotlib.pyplot as plt
import numpy as np

# 数据准备
epochs = list(range(1, 51))
loss_1 = [...]  # 第一项损失
loss_2 = [...]  # 第二项损失

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：堆叠面积图
axes[0].stackplot(epochs, loss_1, loss_2,
                  labels=['Loss 1', 'Loss 2'],
                  alpha=0.7,
                  colors=['#1f77b4', '#ff7f0e'])
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss Value', fontsize=12)
axes[0].set_title('Stacked Loss', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# 右图：分项趋势
axes[1].plot(epochs, loss_1, marker='o', label='Loss 1', linewidth=2)
axes[1].plot(epochs, loss_2, marker='s', label='Loss 2', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss Value', fontsize=12)
axes[1].set_title('Individual Losses', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### 2. 分布对比图

```python
import matplotlib.pyplot as plt
import numpy as np

# 数据
normal_data = [...]  # 正常样本分数
anomaly_data = [...]  # 异常样本分数
threshold = 0.5

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：直方图
axes[0].hist(normal_data, bins=50, alpha=0.7, label='Normal', 
             color='green', density=True, edgecolor='black')
axes[0].hist(anomaly_data, bins=50, alpha=0.7, label='Anomaly',
             color='red', density=True, edgecolor='black')
axes[0].axvline(threshold, color='blue', linestyle='--', linewidth=2,
                label=f'Threshold={threshold:.2f}')
axes[0].set_xlabel('Score', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('Score Distribution', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：箱线图
axes[1].boxplot([normal_data, anomaly_data],
                labels=['Normal', 'Anomaly'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_ylabel('Score', fontsize=12)
axes[1].set_title('Box Plot', fontsize=14, fontweight='bold')
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('distribution.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### 3. ROC 曲线

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 数据
y_true = [...]  # 真实标签
y_scores = [...]  # 预测分数

# 计算 ROC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# 绘图
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, zorder=5,
            label=f'Optimal (thresh={optimal_threshold:.3f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

### 表格模板

#### 1. 参数配置表

| 参数名    | 典型值 | 作用         | 调优方向                      | 影响       |
| :-------- | :----- | :----------- | :---------------------------- | :--------- |
| `param_1` | 64     | 控制模型容量 | 容量不足→增大<br/>过拟合→减小 | 表达能力   |
| `param_2` | 1e-3   | 学习率       | 不收敛→减小<br/>收敛慢→增大   | 训练速度   |
| `param_3` | 0.5    | 权重系数     | 该项不收敛→增大               | 多任务平衡 |

#### 2. 对比分析表

| 维度       | 方案 A     | 方案 B      | 备注     |
| :--------- | :--------- | :---------- | :------- |
| **性能**   | ✅ AUC=0.95 | ❌ AUC=0.87  | A 更优   |
| **速度**   | ❌ 2小时/轮 | ✅ 30分钟/轮 | B 更快   |
| **内存**   | ✅ 4GB      | ❌ 16GB      | A 更省   |
| **易用性** | ❌ 复杂配置 | ✅ 自动调参  | B 更易用 |

#### 3. 问题诊断表

| 现象        | 可能原因   | 解决方案              | 相关章节 |
| :---------- | :--------- | :-------------------- | :------- |
| Loss 不下降 | 学习率过小 | 增大学习率至 1e-3     | §2.3     |
| KL 散度→0   | 后验塌缩   | 调大 k 参数，延缓退火 | §2.4     |
| CUDA OOM    | 批次过大   | 减小 batch_size       | §3.1     |

---

## 质量检查清单

使用以下清单检查文档质量：

### 结构完整性
- [ ] 有清晰的目录（带锚点链接）
- [ ] 章节层次不超过 4 层
- [ ] 每个主要章节有"检查点"或"小结"
- [ ] 提供多条学习路径（理论优先/实践优先/问题驱动）

### 可视化丰富度
- [ ] 至少 5 个 Mermaid 图表
- [ ] 至少 3 个代码可视化示例（matplotlib）
- [ ] 至少 5 个表格
- [ ] 图表类型多样（流程图/序列图/决策树/折线图/箱线图）

### 代码质量
- [ ] 所有代码块指定语言
- [ ] 关键代码有逐行注释
- [ ] 提供完整的可运行示例
- [ ] 有"不稳定 vs 稳定"对比
- [ ] 包含预期输出说明

### 渐进式设计
- [ ] 同一概念出现在至少 2 个层次
- [ ] 从简化示例到完整示例
- [ ] 从高层概览到底层细节
- [ ] 每个检查点有明确的验证任务

### 问题导向
- [ ] 有决策树式问题诊断
- [ ] 常见错误独立章节
- [ ] 提供调试技巧
- [ ] 延伸阅读路径（分层次）

### 写作风格
- [ ] 使用 emoji 增强可读性
- [ ] 关键术语粗体标注
- [ ] 复杂概念有类比
- [ ] 引用块突出核心思想
- [ ] 代码/文件名用反引号

### 可操作性
- [ ] 提供最小化复现配置
- [ ] 所有路径用占位符（便于替换）
- [ ] 命令可直接复制运行
- [ ] 有性能基准或预期结果

---

## 使用建议

### 选择合适的模板版本

| 场景     | 推荐版本 | 组合建议             |
| :------- | :------- | :------------------- |
| 入门教程 | 版本 1   | + 版本 4（可视化）   |
| 技术分析 | 版本 2   | + 版本 5（代码实践） |
| 论文复现 | 版本 3   | + 版本 4（可视化）   |
| API 文档 | 版本 5   | + 版本 1（基础部分） |
| 架构设计 | 版本 2   | + 版本 4（可视化）   |

### 自定义建议

1. **根据受众调整深度**
   - 初学者：版本 1 + 减少数学推导
   - 研究者：版本 3 + 增加理论证明
   - 工程师：版本 5 + 增加性能测试

2. **根据主题调整结构**
   - 算法类：强化"原理层"和"数学推导"
   - 工程类：强化"实践层"和"问题诊断"
   - 系统类：强化"架构层"和"模块交互"

3. **迭代改进**
   - 第一版：快速用版本 1 搭框架
   - 第二版：补充可视化（版本 4 组件）
   - 第三版：深化技术细节（版本 2/3 内容）
   - 第四版：添加实践案例（版本 5 内容）

---

## 附录：元模板

如果需要为特定领域创建新模板，可参考以下元模板结构：

```markdown
# [领域] 文档模板

## 适用场景
[明确的使用场景]

## 核心结构
### 第一部分：[层次1名称]
[该层次的目标和内容要求]

### 第二部分：[层次2名称]
[该层次的目标和内容要求]

### 第三部分：[层次3名称]
[该层次的目标和内容要求]

## 必需组件
- [ ] [组件1]（如：架构图）
- [ ] [组件2]（如：代码示例）
- [ ] [组件3]（如：性能测试）

## 可视化要求
[该领域特定的图表类型]

## 质量标准
[领域特定的质量指标]
```

---

**文档版本**：1.0  
**基于**：DyAD_Visualization_Roadmap.md (v2.0)  
**维护者**：请根据实际使用反馈持续改进  
**贡献方式**：欢迎提交新的模板变体或改进建议
