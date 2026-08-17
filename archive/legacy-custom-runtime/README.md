# Legacy custom runtime

这里归档的是迁移到 OpenAI Agents SDK 之前的自定义实现：

- `agent_loop.py`：手写 Agent Loop
- `llm_router.py`：手写任务路由
- `tool_runtime.py`：手写工具执行器
- `answer_generator.py`：手写回答生成器
- `guard_layer.py`：旧规则护栏
- 对应的旧单元测试和冒烟脚本

当前主入口已经改为：

```text
Agents SDK Agent + Runner + Pydantic + GLM-4V Tool
```
