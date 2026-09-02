# Growth Copilot Eval Methodology

## 评测目标

评测不是回答“模型看起来聪不聪明”，而是判断一次增长分析任务能否安全、正确、可解释地完成。

## 统一分母

`end_to_end_task_success_rate` 使用所有 Golden Case 作为分母。以下任一步失败，整题失败：

1. SQL 生成
2. SQL Guard
3. DuckDB 执行
4. 完整结果等价
5. 指标合规

语义正确率同样以所有有 Golden SQL 的题目为分母，不因执行失败而排除样本。

## 确定性与概率性评测边界

优先使用代码评测：

- `IntentValidator` 的合法指标、参数边界、能力提示与澄清政策
- `Policy Gate` 的工具白名单、能力边界与步数预算
- Agent trajectory 的工具选择、动作顺序、错误恢复和安全停止
- SQL 可执行性
- 表和关键模式合规
- 完整结果等价
- GMV 归因可加和
- 履约 mix/within 与客户州贡献可加和
- 比例范围和窗口互斥

只有洞察相关性、表达清晰度、建议可执行性等缺少唯一标准答案的任务，才考虑 LLM-as-judge；引入前必须用人工样本校准。

意图理解本身使用 DeepSeek JSON Output，因此 8 道 intent case 必须调用真实模型；无 Key 时 fail fast，不用关键词规则伪造意图准确率。模型输出还必须通过本地 `IntentValidator`。

Planner 使用相同的结构化输出原则：模型只能从当前 Tool Registry 提供的动作中提议一个动作，`Policy Gate` 再决定批准或纠正。`eval/trajectory_cases.jsonl` 的 12 条 Case 同时支持两种模式：

- 默认模式验证确定性 Policy 与 LangGraph 控制契约；
- `--use-llm` 模式验证真实模型的工具选择与动作轨迹。

两种结果必须分开报告。Policy 轨迹 12/12 不能被表述为 LLM Planner 准确率。

## 失败分类

```text
intent → planning → policy → tool → guard/execution → validation → insight
```

每个失败保存原问题、SQL、模型/Prompt/指标版本、校验信息和 Trace，便于从失败样本建立回归用例。

## 版本比较

模型、Prompt、Tool Registry 或 Recipe 变更时至少比较：

- 端到端任务成功率
- 完整结果正确率
- 指标合规率
- P50/P95 延迟
- 每个成功任务的成本
- 重试率
- 工具选择正确率
- 轨迹成功率
- Policy Gate 纠正率

质量提升如果伴随延迟或成本上升，应明确记录产品取舍。
