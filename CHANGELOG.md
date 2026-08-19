# Changelog

## 0.3.0 — Agent Harness MVP

### Agent runtime

- 迁移到 OpenAI Agents SDK 的 `Agent + Runner` 运行循环。
- 使用 Pydantic 定义 Agent Context、Tool 参数和结构化结果。
- 保留 DeepSeek OpenAI-compatible 文本模型与 GLM-4V-Flash 视觉模型的 Provider 分离。

### Scope、Capability 和 Tool

- 增加硬规则 Scope Guard 与 DeepSeek 语义 Scope Guard。
- Scope 结果结构化为 `allow / clarify / refuse`、风险等级、原因码和能力 ID。
- 增加 Capability Registry，支持能力到 Tool 的映射。
- 根据当前能力动态暴露 `analyze_current_image` 或 `read_image_metadata`。
- 增加统一 Tool Input Guard、通用 Output Guard 和 Tool 专属 Output Validator。
- 增加工具权限策略：只读工具放行、未来有副作用的工具要求审批、高风险工具拒绝。

### Memory 和 Context

- 使用 Agents SDK `SQLiteSession` 保存会话消息。
- 增加图片引用存储，支持 `image#N`、上一张图片和多图引用。
- 增加 Scope Memory，保存会话目标、风险状态和拒绝主题。
- 增加用户长期记忆 V0：仅保存用户明确要求记住的非敏感内容。
- 增加会话摘要和非破坏式上下文压缩：完整历史保留，模型接收摘要与最近消息。

### Reliability、Trace 和 API

- 增加模型重试策略、网络/超时/HTTP 错误分类和工具超时处理。
- Trace 记录模型、视觉模型、Guard 决策、Tool 调用、Token 和耗时。
- FastAPI 增加会话列表、历史消息、会话删除、图片上传和错误编号。
- API 错误写入 `data/api_errors.jsonl`，并对 Key/Bearer Token 做脱敏。
- 前端支持会话切换、删除、回车发送和图片预览。

### Evaluation

- 增加 Scope 开发集和 Holdout 集。
- 增加 Capability 与 Tool Exposure 评测。
- 增加 40 条摄影专项评测数据结构，当前使用公开 CC0 图片素材。
- 增加数据集校验脚本和 61 个本地测试。

### 尚未包含

- 40 条摄影专项评测的三基线真实对比报告。
- 小红书 Skill/MCP。
- 完整 Human-in-the-loop 审批恢复流程。
- CI、Docker、生产鉴权、限流和多实例部署。
