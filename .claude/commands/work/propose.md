# /work:prop

> 从专家视角生成多个备选方案并进行 trade-off 分析

---

## Usage

```bash
/work:prop "[决策描述]" --expert [类型]
```

### Parameters

- **decision** (required): 决策问题描述
- **--expert**: 专家类型（默认: auto）

---

## Expert Types

| 专家 | 代码 | 聚焦领域 | 对应层级 |
|-----|------|---------|---------|
| System Architect | `architect` | 可扩展性、技术债务、架构权衡 | Digitalization Layer |
| Data Scientist | `data` | 数据质量、模型性能、可解释性 | Data Product Layer |
| AI Strategist | `ai` | AI 战略、转型路径、组织赋能 | AI Transformation Layer |
| `auto` | `auto` | 根据决策内容自动选择 | - |

---

## Examples

```bash
# 系统架构决策
/work:prop "选择数据存储方案" --expert architect

# AI 战略决策
/work:prop "是否引入 LLM API" --expert ai

# 数据建模决策
/work:prop "推荐系统算法选择" --expert data

# 自动选择专家
/work:prop "技术栈选型"
```

---

## Output

1. **专家视角分析**: 从指定专家的角度分析问题
2. **备选方案**: 生成 3-5 个可行的解决方案
3. **Trade-off 矩阵**: 每个方案在不同维度的权衡分析
4. **推荐方案**: 基于分析的最优选择及理由

---

## Workflow

当命令被调用时：

1. **加载专家 Persona**
   - 读取 `.claude/reference/expert-roles/{expert}.md`
   - 采用专家的决策框架和问题风格

2. **专家视角分析**
   - 使用专家的核心关注点分析问题
   - 识别关键权衡维度
   - 评估约束条件和资源

3. **生成备选方案**
   - 3-5 个差异化方案（避免相似方案）
   - 包括保守、激进、平衡选项
   - 考虑短期和长期影响

4. **Trade-off 分析**
   - 按专家框架的关键维度评估
   - 使用表格清晰展示权衡
   - 标注每个方案的主要风险

5. **输出推荐**
   - 基于分析的最优方案
   - 实施建议和注意事项
   - 何时重新评估的建议

---

## Integration

- **Related to**: [quick.md](../../reference/patterns/quick.md) - 参考 Decision Framework patterns
- **Complements**: `/work:dive` - 先深度分析，再生成方案
- **Uses**: `reference/expert-roles/` - 加载专家角色配置

---

**Version**: 1.0
**Last Updated**: 2026-01-22
