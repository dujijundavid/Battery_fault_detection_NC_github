# /work:dive

> 使用 sequential thinking MCP 生成高信息增益问题进行深度分析

---

## Usage

```bash
/work:dive "[主题]" [--questions N]
```

### Parameters

- **topic** (required): 分析主题
- **--questions**: 问题数量 5-10 (默认: 8)

---

## Examples

```bash
# 深度分析架构问题
/work:dive "RTM_Tool 的架构设计瓶颈"

# 分析技术选型
/work:dive "是否使用 TypeScript 重构 Python 项目"

# 生成更多问题
/work:dive "AI 产品化策略" --questions 10
```

---

## Output

1. **Sequential Thinking 分析**: 使用 `mcp__sequential-thinking__sequentialthinking` 进行结构化思考
2. **高信息增益问题**: 生成 5-8 个（或指定数量）能带来最大信息量的问题
3. **问题理由**: 每个问题附说明（为什么这个问题重要，它如何帮助决策）
4. **探索顺序**: 建议问题的探索优先级

---

## Workflow

当命令被调用时：

1. **加载 Sequential Thinking MCP**
   - 使用 `mcp__sequential-thinking__sequentialthinking` 工具
   - 设置思考步骤（通常 5-8 步）

2. **结构化分析主题**
   - 理解问题的核心维度
   - 识别关键不确定性
   - 找出隐含假设

3. **生成高信息增益问题**
   - 每个问题应该能显著减少不确定性
   - 优先考虑二阶/三阶后果的问题
   - 平衡短期战术和长期战略问题

4. **输出建议**
   - 问题按优先级排序
   - 每个问题附"为什么重要"说明
   - 建议探索顺序（哪些问题应该先回答）

---

## Integration

- **Related to**: [quick_patterns.md](../../assets/patterns/quick_patterns.md) - 参考更多分析 patterns
- **Complements**: `/work:prop` - 用于深度分析后的方案生成
- **Uses MCP**: `sequential-thinking` - 结构化推理

---

**Version**: 1.0
**Last Updated**: 2026-01-22
