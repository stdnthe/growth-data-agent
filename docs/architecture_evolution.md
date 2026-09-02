# Architecture Evolution: Single Controller, Extensible Capabilities

## 当前架构决策

Growth Copilot 采用一个 LLM Controller，而不是把取数、归因、图表和报告拆成多个相互对话的 LLM Agent。

```text
User Question
  -> Structured Intent + IntentValidator
  -> ContextBundle
  -> CapabilityRegistry selects relevant capabilities
  -> Single AgentController selects one allowed action
  -> Policy Gate
  -> Deterministic Tool / Recipe
  -> Validator returns an observation
  -> continue / finish / clarify / safe stop
```

原因不是否定 Multi-Agent，而是当前只有一个 Olist 交易语义域、三个边界明确的分析能力。拆分多个 LLM 会增加上下文交接、延迟、成本和轨迹评估难度，却没有证据证明能提升结果正确率。

LangGraph 在本项目中负责状态与有限循环，不等于 Multi-Agent。`AgentController` 是唯一的模型决策者；SQL Guard、GMV/履约 Recipe 和 Validator 都是受控工具或确定性程序。

## 当前已注册能力

| Capability | 业务任务 | 允许动作 | 输出 Artifact | 关键校验 |
| --- | --- | --- | --- | --- |
| `metric_query` | 指标问数、趋势、对比 | `query_metric` | `ValidatedQueryResult` | SQL Guard、输出契约、时间/比例检查 |
| `gmv_diagnosis` | 解释 GMV 变化 | `run_gmv_recipe`、`query_metric` | `GmvAttributionReport` | Orders × AOV 分解、维度贡献可加和 |
| `fulfillment_diagnosis` | 解释履约指标变化 | `run_fulfillment_recipe`、`query_metric` | `FulfillmentDiagnosisReport` | mix/within 与州贡献可加和 |

`CapabilityRegistry` 只向 Controller 暴露当前意图相关的少量动作。漏斗和预测尚未注册，因此模型当前不能选择或声称已经支持它们。

## 新能力接入合同

任何新功能都必须先成为一个可独立验收的 Capability，而不是先给模型增加一个工具名称。

1. **业务问题**：用户要做什么决策，而不只是算法名称。
2. **数据前提**：需要哪些事实、事件、身份和时间字段。
3. **输入契约**：指标、粒度、维度、窗口和必要参数。
4. **执行引擎**：SQL、确定性 Recipe 或统计模型；LLM 不直接计算结果。
5. **输出 Artifact**：可被 Validator、图表和报告稳定消费的结构化对象。
6. **Validator**：正确性、范围、证据充分性和安全停止条件。
7. **Eval**：单能力、工具交接和端到端轨迹 Case。
8. **产品呈现**：结论、证据、限制与可复现元数据。

只有完成以上合同，才在 `CapabilityRegistry` 注册并向 Controller 暴露。

## 漏斗分析的未来扩展

当前 Olist 没有曝光、点击、浏览、加购和广告成本事件，不能实现真实营销漏斗，也不能声称支持 CTR、弃购率、CAC、ROAS 或广告归因。

可以先设计但暂不实现两种不同能力：

- `order_journey_analysis`：基于订单创建、支付确认、交付承运商、签收和评价时间戳分析履约链路。它不是营销转化漏斗。
- `behavior_funnel_analysis`：只有接入真实行为事件后才启用；输入契约必须包含主体 ID、事件定义、步骤顺序、去重规则、转化窗口和分母。

关键 Validator 包括身份一致性、时间顺序、窗口覆盖、Join 唯一性和步骤计数规则。

## 时序预测的未来扩展

预测能力应注册为 `forecast_metric` 工具，而不是新增一个 Forecast LLM Agent。模型负责理解预测目标和解释结果，数值由统计程序生成。

建议执行链：

```text
validate history
  -> build regular time series
  -> seasonal-naive baseline
  -> candidate model
  -> rolling backtest
  -> compare with baseline
  -> forecast + interval or safe refusal
```

第一版只考虑 GMV、订单量或履约时长的短期日/周预测。必须记录预测期、训练窗口、回测误差、基线结果和预测区间。若历史不足或候选模型未优于基线，产品应展示基线或拒绝给出伪精确预测。

Olist 是历史静态数据，因此它适合作为回测和产品能力演示，不代表已经具备线上实时预测价值。

## 什么时候才升级 Multi-Agent

至少出现一种有证据的结构性需求后再考虑：

- 多个数据域具有不同权限、上下文或长期记忆；
- 子任务可以真正并行，并能显著降低端到端时延；
- 每个子任务都能独立产出可验证 Artifact；
- 单一 Controller 的上下文或工具选择已经成为可测量的质量瓶颈。

如果多个组件使用相同模型、相同上下文和相同权限，只是串行传递文本，它们更适合作为工具或 Workflow，而不是独立 Agent。

## 面试回答

> 我没有为了展示前沿概念把系统刻意拆成 Multi-Agent。当前项目只有一个受控的 Olist 交易语义域，所以我选择单一 LLM Controller，通过 Capability Registry 只暴露与当前问题相关的少量工具。LangGraph 管理有限循环，模型选择动作，但 SQL Guard、确定性 Recipe、Validator 和步数预算决定动作能否执行以及何时停止。
>
> 这个架构仍然可以扩展。未来增加漏斗或预测时，我会把它们作为有数据合同、输出 Artifact、Validator 和专属 Eval 的 Capability 接入，而不是增加一个会自由对话的子 Agent。真正的行为漏斗需要新的事件数据域；时序预测需要滚动回测、基线和预测区间。只有当数据域权限分离、任务可并行，或者单一 Controller 出现可测量瓶颈时，我才会引入 Multi-Agent。

这段回答区分了已实现能力、计划扩展和采用 Multi-Agent 的触发条件，避免把架构复杂度本身当作产品价值。
