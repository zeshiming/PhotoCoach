"""基于 OpenAI Agents SDK 的 PhotoCoach 主运行时。

当前版本使用：

* Agents SDK 的 Agent + Runner 管理 Agent Loop；
* Pydantic 校验工具参数和运行上下文；
* DeepSeek 的 OpenAI-compatible Chat Completions 负责主 Agent；
* GLM-4V-Flash 作为图片分析工具。

由于 DeepSeek/GLM 不是 OpenAI Responses Provider，必须显式使用
OpenAIChatCompletionsModel，并关闭 Agents SDK 默认的 Responses 路径。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

from agents import (
    Agent,
    AsyncOpenAI,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OpenAIChatCompletionsModel,
    ModelRetrySettings,
    ModelSettings,
    RunContextWrapper,
    Runner,
    SQLiteSession,
    input_guardrail,
    function_tool,
    set_tracing_disabled,
)
from pydantic import BaseModel, ConfigDict, Field

from .llm_provider import LLMConfig
from .image_session_store import ImageSessionStore
from .models import AgentRequest
from .trace_store import JsonlTraceStore, TraceRecord
from .reliability import RetryTracker, build_retry_policy
from .vision_provider import GLMVisionProvider


class PhotoAgentContext(BaseModel):
    """传给 SDK 工具的本地运行上下文。

    Context 不会自动发送给 LLM，只供工具函数读取本地图片和 Provider。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    image_store: ImageSessionStore
    active_image_id: str | None = None
    available_image_ids: list[str] = Field(default_factory=list)
    analysis_done: bool = False
    vision_provider: GLMVisionProvider | None = None


class ImageAnalysisArgs(BaseModel):
    """图片分析工具的结构化参数，由 Pydantic 生成 JSON Schema。"""

    focus: str = Field(
        default="构图、光线和曝光",
        description="希望重点分析的摄影维度",
    )
    image_id: str | None = Field(
        default=None,
        description="要分析的单张图片引用，例如 image#1；不填则使用当前图片",
    )
    image_ids: list[str] = Field(
        default_factory=list,
        description="需要分别分析的图片引用列表，用于多图比较，例如 ['image#1', 'image#2']",
    )


def _input_text(value: Any) -> str:
    """从 SDK 的字符串或输入项列表中提取用户文字。"""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") in {
                            "input_text",
                            "text",
                        }:
                            parts.append(str(block.get("text", "")))
        return " ".join(part for part in parts if part).strip()
    return str(value).strip()


@input_guardrail(run_in_parallel=False)
async def photo_input_guard(
    context: RunContextWrapper[PhotoAgentContext],
    agent: Agent[PhotoAgentContext],
    input: Any,
) -> GuardrailFunctionOutput:
    """在 Agent 运行前执行空输入、越界和敏感请求检查。"""

    text = _input_text(input)
    has_any_image = bool(
        context.context.active_image_id or context.context.available_image_ids
    )
    has_resolved_image = bool(context.context.active_image_id)
    sensitive = ("真实身份", "身份证", "几岁", "年龄", "是不是未成年", "人脸识别")
    out_of_scope = ("写简历", "写代码", "股票", "法律咨询", "算命")
    # 只拦截明确的指代引用；“照片中的人脸怎么拍好看”可能是泛化知识问题，
    # 不能仅凭“照片中”三个字误判为缺少图片。
    image_references = ("这张照片", "这张图", "图片里", "这张片子")

    reason: str | None = None
    if not text and not has_any_image:
        reason = "请先输入摄影问题，或说明希望如何分析图片。"
    elif not text and has_any_image:
        reason = "图片已收到，请说明你希望分析构图、光线、曝光还是后期。"
    elif has_any_image and any(item in text for item in sensitive):
        reason = "不能根据图片推断身份、年龄等敏感个人信息。"
    elif any(item in text for item in out_of_scope):
        reason = "当前 PhotoCoach 只处理摄影相关任务。"
    elif not has_resolved_image and any(item in text for item in image_references):
        reason = "图片引用不明确或当前会话没有对应图片，请指定 image#N。"

    return GuardrailFunctionOutput(
        output_info={"reason": reason} if reason else {"reason": "allow"},
        tripwire_triggered=reason is not None,
    )


def vision_tool_enabled(
    context: RunContextWrapper[PhotoAgentContext],
    agent: Agent[PhotoAgentContext],
) -> bool:
    """每轮最多允许一次图片分析工具调用，避免模型重复调用。"""

    return bool(
        context.context.active_image_id or context.context.available_image_ids
    ) and not context.context.analysis_done


@function_tool(
    is_enabled=vision_tool_enabled,
    timeout=60.0,
    timeout_behavior="error_as_result",
)
async def analyze_current_image(
    context: RunContextWrapper[PhotoAgentContext],
    args: ImageAnalysisArgs,
) -> str:
    """分析当前会话中指定或当前引用的图片，并返回视觉观察结果。"""

    state = context.context
    target_ids = args.image_ids or ([args.image_id] if args.image_id else [])
    if not target_ids and state.active_image_id:
        target_ids = [state.active_image_id]
    if not target_ids and len(state.available_image_ids) == 1:
        target_ids = state.available_image_ids
    if not target_ids:
        return "当前请求没有可分析的图片。"
    if state.vision_provider is None:
        return "视觉 Provider 尚未配置。"

    # 先标记，避免工具异常或模型重复规划导致同一轮不断重试。
    state.analysis_done = True

    prompt = (
        f"请重点分析：{args.focus}。"
        "只描述图片中可以观察到的摄影事实，并给出原因。"
        "不要猜测无法从图片确认的相机参数。"
    )
    observations: list[str] = []
    for image_id in target_ids:
        record = state.image_store.get(state.session_id, image_id)
        if record is None:
            observations.append(f"{image_id}：找不到图片引用。")
            continue
        if not record.ref.startswith(("http://", "https://", "data:")):
            if not Path(record.ref).expanduser().is_file():
                observations.append(f"{image_id}：图片文件已不存在：{record.ref}")
                continue

        # GLM-4V-Flash 当前按单图调用；多图比较先逐张分析，再由主 Agent 汇总。
        observation = await asyncio.to_thread(
            state.vision_provider.analyze,
            record.ref,
            prompt,
        )
        observations.append(f"[{image_id}]\n{observation}")
    return "\n\n".join(observations)


class AgentRunResult(BaseModel):
    """SDK 一次运行的对外结果，显式带回 session_id。"""

    session_id: str
    final_output: str
    trace_id: str


def build_photo_agent(
    llm_config: LLMConfig,
    retry_tracker: RetryTracker | None = None,
) -> Agent[PhotoAgentContext]:
    """创建使用 OpenAI-compatible Chat Completions 的 SDK Agent。"""

    # 没有 OpenAI tracing Key 时关闭默认 Trace 上传，避免 SDK 额外请求 OpenAI。
    set_tracing_disabled(True)
    client = AsyncOpenAI(
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        timeout=llm_config.timeout,
        max_retries=llm_config.max_retries,
    )
    model = OpenAIChatCompletionsModel(
        model=llm_config.model,
        openai_client=client,
    )
    tracker = retry_tracker or RetryTracker()
    retry = ModelRetrySettings(
        max_retries=min(llm_config.max_retries, 2),
        backoff={
            "initial_delay": 0.5,
            "max_delay": 5.0,
            "multiplier": 2.0,
            "jitter": True,
        },
        policy=build_retry_policy(tracker),
    )

    return Agent(
        name="PhotoCoach",
        instructions=(
            "你是 PhotoCoach 摄影助手。"
            "处理摄影知识、拍摄建议和图片分析。"
            "如果用户提供了图片并询问图片内容，必须先调用 analyze_current_image。"
            "工具返回结果后，再用中文给出最终回答。"
            "区分图片观察事实和摄影建议，不要编造 EXIF。"
        ),
        model=model,
        model_settings=ModelSettings(
            temperature=0.2,
            parallel_tool_calls=False,
            timeout=llm_config.timeout,
            retry=retry,
            extra_body=(
                {"thinking": {"type": "disabled"}}
                if llm_config.model.startswith("deepseek-")
                else None
            ),
        ),
        tools=[analyze_current_image],
        input_guardrails=[photo_input_guard],
    )


async def run_photo_agent(request: AgentRequest) -> AgentRunResult:
    """运行一次 Agents SDK Agent。"""

    started_at = time.perf_counter()
    trace_id = uuid.uuid4().hex
    trace_store = JsonlTraceStore()
    retry_tracker = RetryTracker()
    llm_config = LLMConfig.from_env()
    session_id = request.session_id or "default"
    db_path = Path("data/agent_sessions.db")
    image_store = ImageSessionStore(db_path)

    # 新图片先注册为会话资产；没有新图片时，尝试解析对上一张图片的引用。
    active_image_id = None
    available_image_ids: list[str] = []
    if request.images:
        turn_no = image_store.next_turn_no(session_id)
        for image_ref in request.images:
            available_image_ids.append(
                image_store.register(session_id, image_ref, turn_no)
            )
        # 单图可直接成为当前图片；多图必须通过明确引用或比较语义选择。
        if len(available_image_ids) == 1:
            active_image_id = available_image_ids[0]
        else:
            active_image_id = image_store.resolve_reference(session_id, request.text)
    else:
        active_image_id = image_store.resolve_reference(session_id, request.text)
        if active_image_id:
            available_image_ids = [item.image_id for item in image_store.list(session_id)]

    # 第二轮没有新图片时，仍需重新创建视觉 Provider 来分析已引用的图片。
    vision_provider = GLMVisionProvider.from_env() if active_image_id else None
    session = SQLiteSession(session_id, str(db_path))
    context = PhotoAgentContext(
        session_id=session_id,
        image_store=image_store,
        active_image_id=active_image_id,
        available_image_ids=available_image_ids,
        vision_provider=vision_provider,
    )
    agent = build_photo_agent(llm_config, retry_tracker)

    input_text = request.text
    if input_text.strip() and (request.images or active_image_id):
        available_ids = ", ".join(
            item.image_id for item in image_store.list(session_id)
        )
        input_text += (
            "\n用户已提供或引用图片，如需分析请调用图片工具。"
            f"当前图片：{active_image_id or '未解析'}。"
            f"可用图片引用：{available_ids}。"
        )

    try:
        result = await Runner.run(
            agent,
            input=input_text,
            context=context,
            session=session,
            max_turns=4,
        )
    except InputGuardrailTripwireTriggered as exc:
        info = exc.guardrail_result.output.output_info
        final_output = str(info.get("reason", "请求未通过输入护栏。"))
        trace_store.append(
            TraceRecord(
                trace_id=trace_id,
                session_id=session_id,
                model=llm_config.model,
                vision_model=vision_provider.config.model if vision_provider else None,
                status="blocked",
                duration_ms=int((time.perf_counter() - started_at) * 1000),
                retry_count=retry_tracker.retry_count,
                failure_type=retry_tracker.failure_type,
                error=final_output,
            )
        )
        return AgentRunResult(
            session_id=session_id,
            final_output=final_output,
            trace_id=trace_id,
        )

    tool_calls: list[dict[str, Any]] = []
    for item in result.new_items:
        raw_item = getattr(item, "raw_item", None)
        raw_type = (
            raw_item.get("type")
            if isinstance(raw_item, dict)
            else getattr(raw_item, "type", None)
        )
        raw_name = (
            raw_item.get("name")
            if isinstance(raw_item, dict)
            else getattr(raw_item, "name", None)
        )
        if raw_type in {"function_call", "computer_call"}:
            tool_calls.append(
                {
                    "name": raw_name or raw_type,
                    "image_id": active_image_id,
                }
            )

    usage_data: dict[str, Any] = {}
    usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
    if usage is not None:
        usage_data = {
            "requests": getattr(usage, "requests", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    trace_store.append(
        TraceRecord(
            trace_id=trace_id,
            session_id=session_id,
            model=llm_config.model,
            vision_model=vision_provider.config.model if vision_provider else None,
            status="success",
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            retry_count=retry_tracker.retry_count,
            failure_type=retry_tracker.failure_type,
            tool_calls=tool_calls,
            usage=usage_data,
        )
    )
    return AgentRunResult(
        session_id=session_id,
        final_output=str(result.final_output),
        trace_id=trace_id,
    )
