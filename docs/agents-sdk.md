# PhotoCoach：OpenAI Agents SDK 架构

## 当前主运行时

```text
Agents SDK Agent
  ↓
Runner
  ↓
Pydantic Tool Schema
  ↓
GLM-4V-Flash 图片工具
  ↓
最终回答
```

## Provider 分工

- DeepSeek：主 Agent 的文本任务理解和回答。
- GLM-4V-Flash：通过 `analyze_current_image` Function Tool 分析图片。
- OpenAI Agents SDK：负责 Agent Loop、工具调用和结果回传。
- Pydantic：校验工具参数和运行上下文。

## 为什么使用 Chat Completions Model

DeepSeek 和智谱接口是 OpenAI-compatible Chat Completions 接口，不是 OpenAI Responses 接口。因此使用 `OpenAIChatCompletionsModel`，避免 SDK 默认调用不兼容的 Responses 路径。

## 当前入口

```bash
PYTHONPATH=. .venv/bin/python scripts/run_agent.py \
  --text "请分析这张照片的构图、光线和曝光，并给出具体建议。" \
  --image "/path/to/photo.jpg"
```

## 多轮图片会话

- `SQLiteSession` 保存 SDK 对话项、工具调用和工具结果。
- `ImageSessionStore` 保存 `image#N`、本地路径/URL、轮次和文件 Hash。
- 第二轮没有新图片时，通过 `session_id` 和自然语言引用恢复上一张图片。
- 多图比较通过工具参数 `image_ids` 逐张分析，再由主 Agent 汇总。

## 本地 Trace

每次运行会追加写入 `data/traces.jsonl`，记录 `trace_id`、`session_id`、模型、工具调用、耗时、Token 使用和错误状态，不记录 API Key 或图片像素。

## 旧实现

旧的自定义 Agent Loop、ToolExecutor、Router 和测试已移动到 `archive/legacy-custom-runtime/`，用于理解迁移前后的差异。主 CLI 使用 `agents_sdk_app.py`。
