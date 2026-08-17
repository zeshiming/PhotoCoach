# PhotoCoach V0.2：任务理解与能力路由

## 目标

PhotoCoach 不再把固定的 6 个 Intent 作为核心路由。LLM 先理解用户目标，再输出所需能力；工具注册表负责提供能力的具体实现。

```text
输入 → 规则护栏 → TaskUnderstanding → 能力召回 → 工具选择 → Agent Loop
```

## 核心概念

- `TaskUnderstanding`：LLM 对用户目标、图片需求、风险和澄清状态的结构化理解。
- `CapabilitySpec`：稳定的能力契约，例如 `image_understanding`、`web_search`。
- `ToolSpec`：具体工具实现，例如某个视觉模型或网页搜索 API。
- `ExecutionPlan`：由步骤组成的可校验执行计划。
- 不设置固定的 Intent 枚举；需要统计时直接记录 `goal` 和 `capabilities`。

## 为什么能力优先

同一个能力可以有多个工具实现，工具也可能随着部署环境变化。把能力和工具分开后，新增工具不需要修改意图分类器。

```text
image_understanding
  ├── vision_model_a
  └── vision_model_b
```

## LLM 输出约束

LLM 必须输出 `TaskUnderstanding` 对应的 JSON。进入 Agent Loop 前必须校验：

1. `goal` 非空；
2. 所有能力名称存在于注册表；
3. 需要澄清时必须提供问题；
4. OOS 请求不能被标记为普通回答；
5. 计划中的工具调用必须提供工具名。

## 后续演进

当工具数量较少时，可以把能力注册表直接提供给 LLM。当工具数量增加时，再加入关键词/Embedding 召回和 LLM 重排。Embedding 的职责是召回候选能力或工具，而不是把所有用户请求强行分类成固定 Intent。
