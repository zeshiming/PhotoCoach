# PhotoCoach

一个支持纯文字咨询和图片对话的单 Agent 摄影助手。

当前阶段：v0.3.0 OpenAI Agents SDK + Pydantic + 多模态工具。

## 目标

用户可以直接用文字咨询摄影问题，也可以上传照片；PhotoCoach 结合文字、图片和对话上下文，给出具体、可执行、符合用户设备与拍摄目标的摄影建议。

## 当前文档

- `docs/spec.md`：V0 产品范围、边界和成功标准
- `docs/task-routing-v2.md`：任务理解、能力注册表和路由主线
- `docs/agent-loop.md`：后续工具执行循环设计
- `CHANGELOG.md`：当前版本的详细变更说明

## 当前运行主线

```text
Scope Guard → Capability Registry → Agents SDK Agent + Runner → 动态 Tool Exposure → GLM-4V / 元数据工具
```

真实模型配置位于本地 `.env`，不提交 API Key。旧的关键词和 embedding 路由保存在 `archive/legacy-routing-v0/`，仅用于回顾和对比。底层 Provider 代码仍保留，用于多 Provider 适配和对照测试。

## 启动 FastAPI 和前端

```bash
cd ~/Desktop/PhotoCoach
PYTHONPATH=. .venv/bin/uvicorn photo_coach.api:app --reload --port 8001
```

打开：<http://127.0.0.1:8001>

接口：

- `GET /health`：健康检查
- `POST /api/v1/chat`：文字和图片对话，表单字段为 `message`、`session_id`、`image`
- `GET /docs`：FastAPI 自动生成的接口文档

## 评测

```bash
# 校验摄影专项评测集和公开图片引用
PYTHONPATH=. .venv/bin/python eval/validate_photo_eval.py --image-root eval

# Scope / Capability 评测
PYTHONPATH=. .venv/bin/python eval/run_scope_eval.py --mode mock --strict
PYTHONPATH=. .venv/bin/python eval/run_capability_eval.py --mode mock --strict
```

摄影专项评测数据位于 `eval/photo_quality_eval.jsonl`，图片来源和授权记录位于 `eval/assets/SOURCES.md`。三条真实基线的对比报告尚未生成，作为下一阶段工作。
