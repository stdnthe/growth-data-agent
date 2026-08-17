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

- 路由与参数精确匹配
- SQL 可执行性
- 表和关键模式合规
- 完整结果等价
- GMV 归因可加和
- 比例范围和窗口互斥

只有洞察相关性、表达清晰度、建议可执行性等缺少唯一标准答案的任务，才考虑 LLM-as-judge；引入前必须用人工样本校准。

## 失败分类

```text
intent → generation → guard → execution → validation → insight
```

每个失败保存原问题、SQL、模型/Prompt/指标版本、校验信息和 Trace，便于从失败样本建立回归用例。

## 版本比较

模型、Prompt 或 Workflow 变更时至少比较：

- 端到端任务成功率
- 完整结果正确率
- 指标合规率
- P50/P95 延迟
- 每个成功任务的成本
- 重试率

质量提升如果伴随延迟或成本上升，应明确记录产品取舍。
