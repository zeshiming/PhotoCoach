# PhotoCoach

一个支持纯文字咨询和图片对话的单 Agent 摄影助手。

当前阶段：V0.3 OpenAI Agents SDK + Pydantic + 多模态工具。

## 目标

用户可以直接用文字咨询摄影问题，也可以上传照片；PhotoCoach 结合文字、图片和对话上下文，给出具体、可执行、符合用户设备与拍摄目标的摄影建议。

## 当前文档

- `docs/spec.md`：V0 产品范围、边界和成功标准
- `docs/task-routing-v2.md`：任务理解、能力注册表和路由主线
- `docs/agent-loop.md`：后续工具执行循环设计

## 当前运行主线

```text
Agents SDK Agent + Runner → Pydantic Tool Schema → GLM-4V 图片工具
```

真实模型配置位于本地 `.env`，不提交 API Key。旧的关键词和 embedding 路由保存在 `archive/legacy-routing-v0/`，仅用于回顾和对比。底层 Provider 代码仍保留，用于多 Provider 适配和对照测试。
