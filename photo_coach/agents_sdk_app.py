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
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image
from agents import (
    Agent,
    AsyncOpenAI,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailTripwireTriggered,
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
from agents.decorators import tool_input_guardrail, tool_output_guardrail
from pydantic import BaseModel, ConfigDict, Field

from .llm_provider import LLMConfig
from .image_session_store import ImageSessionStore
from .models import AgentRequest
from .trace_store import JsonlTraceStore, TraceRecord
from .reliability import RetryTracker, build_retry_policy
from .scope_models import ScopeDecision, ScopeResult
from .scope_guard import SemanticScopeGuard, ScopeGuardError
from .scope_memory import ScopeMemoryStore
from .scope_policy import DEFAULT_SCOPE_POLICY, ScopePolicy
from .capability_registry import (
    DEFAULT_CAPABILITY_REGISTRY,
    CapabilityRegistry,
)
from .context_compaction import CompactingSession, ContextCompactor
from .memory_store import LongTermMemoryStore
from .tool_permissions import (
    DEFAULT_TOOL_PERMISSION_POLICY,
    PermissionDecision,
    ToolPermissionPolicy,
)
from .vision_provider import GLMVisionProvider


class PhotoAgentContext(BaseModel):
    """传给 SDK 工具的本地运行上下文。

    Context 不会自动发送给 LLM，只供工具函数读取本地图片和 Provider。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    image_store: ImageSessionStore
    request_text: str = ""
    active_image_id: str | None = None
    available_image_ids: list[str] = Field(default_factory=list)
    analysis_done: bool = False
    metadata_done: bool = False
    vision_provider: GLMVisionProvider | None = None
    scope_guard: SemanticScopeGuard | None = None
    guard_model: str | None = None
    guard_checks: list[dict[str, Any]] = Field(default_factory=list)
    scope_memory: ScopeMemoryStore | None = None
    scope_policy: ScopePolicy = DEFAULT_SCOPE_POLICY
    capability_registry: CapabilityRegistry = DEFAULT_CAPABILITY_REGISTRY
    selected_capabilities: list[str] = Field(default_factory=list)
    user_id: str = "default-user"
    long_term_memory: LongTermMemoryStore | None = None
    session_summary: str | None = None
    permission_policy: ToolPermissionPolicy = DEFAULT_TOOL_PERMISSION_POLICY


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


class ImageMetadataArgs(BaseModel):
    """图片元数据工具的结构化参数。"""

    image_id: str | None = Field(
        default=None,
        description="要读取的图片引用，例如 image#1；不填则使用当前图片",
    )


class ImageMetadataResult(BaseModel):
    """元数据工具的结构化输出；失败也必须使用同一个 Schema。"""

    ok: bool
    image_id: str | None = None
    width: int | None = None
    height: int | None = None
    format: str | None = None
    mode: str | None = None
    exif: dict[str, object] = Field(default_factory=dict)
    error: str | None = None


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


def _record_guard_check(
    state: PhotoAgentContext,
    *,
    stage: str,
    layer: str,
    result: ScopeResult,
    duration_ms: int,
) -> None:
    """把 Guard 的最小审计信息写入本次运行上下文。"""

    state.guard_checks.append(
        {
            "stage": stage,
            "layer": layer,
            "decision": result.decision.value,
            "scope": result.scope,
            "risk_level": result.risk_level,
            "reason_code": result.reason_code,
            "capabilities": list(result.capabilities),
            "duration_ms": duration_ms,
            "model": state.guard_model if layer == "semantic" else None,
        }
    )


def _select_capabilities(state: PhotoAgentContext, result: ScopeResult) -> None:
    """校验模型返回的能力，并保存当前轮的动态工具选择。"""

    if result.decision is not ScopeDecision.ALLOW:
        state.selected_capabilities = []
        return
    selected = state.capability_registry.validate(result.capabilities)
    if not selected:
        selected = state.capability_registry.infer(state.request_text)
    state.selected_capabilities = selected


def _record_scope_memory(state: PhotoAgentContext, result: ScopeResult) -> None:
    """保存最新 Guard 决策，但不保存模型思维链。"""

    if state.scope_memory is not None:
        state.scope_memory.record(
            state.session_id,
            result,
            goal=state.request_text,
            last_image_id=state.active_image_id,
        )


def _scope_context_summary(state: PhotoAgentContext) -> str:
    """组合图片上下文和跨轮 Scope Memory 摘要。"""

    memory = (
        state.scope_memory.summary(state.session_id)
        if state.scope_memory is not None
        else "当前会话还没有 Scope Memory。"
    )
    long_term = (
        state.long_term_memory.context_text(state.user_id, state.request_text)
        if state.long_term_memory is not None
        else "暂无用户长期记忆。"
    )
    return (
        f"{memory}；用户长期记忆：{long_term}；"
        f"会话摘要：{state.session_summary or '暂无'}；"
        f"当前会话图片：{state.active_image_id or '无'}；"
        f"可用图片：{', '.join(state.available_image_ids) or '无'}"
    )


def hard_rule_check(
    text: str,
    state: PhotoAgentContext,
    policy: ScopePolicy = DEFAULT_SCOPE_POLICY,
) -> ScopeResult:
    """执行零成本硬规则，不调用模型。"""

    has_any_image = bool(state.active_image_id or state.available_image_ids)
    has_resolved_image = bool(state.active_image_id)
    if not text and not has_any_image:
        return ScopeResult(
            decision=ScopeDecision.CLARIFY,
            scope="ambiguous",
            risk_level="low",
            reason_code="empty_input",
            message="请先输入摄影问题，或说明希望如何分析图片。",
        )
    if not text and has_any_image:
        return ScopeResult(
            decision=ScopeDecision.CLARIFY,
            scope="ambiguous",
            risk_level="low",
            reason_code="image_without_request",
            message="图片已收到，请说明你希望分析构图、光线、曝光还是后期。",
        )
    if has_any_image and any(item in text for item in policy.sensitive_topics):
        return ScopeResult(
            decision=ScopeDecision.REFUSE,
            scope="restricted",
            risk_level="high",
            reason_code="sensitive_inference",
            message="不能根据图片推断身份、年龄等敏感个人信息。",
        )
    if any(item in text for item in policy.out_of_scope_topics):
        return ScopeResult(
            decision=ScopeDecision.REFUSE,
            scope="out_of_scope",
            risk_level="low",
            reason_code="non_photography_task",
            message="当前 PhotoCoach 只处理摄影相关任务。",
        )
    if not has_resolved_image and any(
        item in text for item in policy.image_reference_markers
    ):
        return ScopeResult(
            decision=ScopeDecision.CLARIFY,
            scope="ambiguous",
            risk_level="low",
            reason_code="missing_image_reference",
            message="图片引用不明确或当前会话没有对应图片，请指定 image#N。",
        )
    return ScopeResult(
        decision=ScopeDecision.ALLOW,
        scope="in_scope",
        risk_level="low",
        reason_code="photography_request",
        message="allow",
    )


async def semantic_scope_check(text: str, state: PhotoAgentContext) -> ScopeResult:
    """调用语义 Guard；Provider 异常时返回安全的澄清结果。"""

    if state.scope_guard is None:
        return ScopeResult(
            decision=ScopeDecision.ALLOW,
            scope="in_scope",
            risk_level="low",
            reason_code="scope_guard_disabled",
            message="allow",
        )
    try:
        return await state.scope_guard.evaluate(
            text,
            context_summary=_scope_context_summary(state),
        )
    except ScopeGuardError:
        return ScopeResult(
            decision=ScopeDecision.CLARIFY,
            scope="ambiguous",
            risk_level="unknown",
            reason_code="scope_guard_unavailable",
            message="暂时无法确认请求范围，请换一种摄影问题描述。",
        )


@input_guardrail(run_in_parallel=False)
async def photo_input_guard(
    context: RunContextWrapper[PhotoAgentContext],
    agent: Agent[PhotoAgentContext],
    input: Any,
) -> GuardrailFunctionOutput:
    """编排硬规则和语义检查，并把结果交给 SDK。"""

    state = context.context
    text = _input_text(input)
    hard_started = time.perf_counter()
    scope_result = hard_rule_check(text, state, state.scope_policy)
    _record_guard_check(
        state,
        stage="input",
        layer="hard_rule",
        result=scope_result,
        duration_ms=int((time.perf_counter() - hard_started) * 1000),
    )
    _select_capabilities(state, scope_result)
    _record_scope_memory(state, scope_result)

    if not scope_result.should_block and state.scope_guard is not None:
        semantic_started = time.perf_counter()
        scope_result = await semantic_scope_check(text, state)
        _record_guard_check(
            state,
            stage="input",
            layer="semantic",
            result=scope_result,
            duration_ms=int((time.perf_counter() - semantic_started) * 1000),
        )
        _select_capabilities(state, scope_result)
        _record_scope_memory(state, scope_result)

    output_info = scope_result.model_dump(mode="json")
    output_info["reason"] = scope_result.message
    return GuardrailFunctionOutput(
        output_info=output_info,
        tripwire_triggered=scope_result.should_block,
    )


def vision_tool_enabled(
    context: RunContextWrapper[PhotoAgentContext],
    agent: Agent[PhotoAgentContext],
) -> bool:
    """每轮最多允许一次图片分析工具调用，避免模型重复调用。"""

    capability_allowed = not context.context.selected_capabilities or "analyze_image" in context.context.selected_capabilities
    return capability_allowed and bool(
        context.context.active_image_id or context.context.available_image_ids
    ) and not context.context.analysis_done


def metadata_tool_enabled(
    context: RunContextWrapper[PhotoAgentContext],
    agent: Agent[PhotoAgentContext],
) -> bool:
    """只有存在图片且本轮尚未读取元数据时才开放元数据工具。"""

    capability_allowed = not context.context.selected_capabilities or "read_metadata" in context.context.selected_capabilities
    return capability_allowed and bool(
        context.context.active_image_id or context.context.available_image_ids
    ) and not context.context.metadata_done


def make_tool_needs_approval(tool_name: str):
    """为固定工具名生成 Agents SDK 审批回调。

    SDK 回调的第三个参数是 call_id，不是工具名，所以需要在注册工具时闭包绑定名称。
    """

    async def _needs_approval(
        context: RunContextWrapper[PhotoAgentContext],
        args: dict[str, Any],
        call_id: str,
    ) -> bool:
        return (
            context.context.permission_policy.decision(tool_name)
            is PermissionDecision.ASK
        )

    return _needs_approval


@tool_input_guardrail
async def photo_tool_scope_guard(data: Any) -> ToolGuardrailFunctionOutput:
    """在图片工具真正执行前检查工具名、用户请求和工具参数。"""

    guard_started = time.perf_counter()
    tool_context = data.context
    state = tool_context.context
    permission = state.permission_policy.decision(tool_context.tool_name)
    if permission is PermissionDecision.DENY:
        return ToolGuardrailFunctionOutput.raise_exception(
            output_info={
                "decision": "deny",
                "reason_code": "tool_permission_denied",
                "message": state.permission_policy.reason(tool_context.tool_name),
            }
        )
    try:
        tool_arguments = json.loads(tool_context.tool_arguments or "{}")
    except json.JSONDecodeError:
        result = ScopeResult(
            decision=ScopeDecision.REFUSE,
            scope="ambiguous",
            risk_level="high",
            reason_code="invalid_tool_arguments",
            message="工具参数格式无效，已阻止执行。",
        )
        _record_guard_check(
            state,
            stage="tool",
            layer="hard_rule",
            result=result,
            duration_ms=int((time.perf_counter() - guard_started) * 1000),
        )
        _record_scope_memory(state, result)
        return ToolGuardrailFunctionOutput.raise_exception(
            output_info=result.model_dump(mode="json")
        )

    if state.scope_guard is None:
        # 离线单测或显式关闭语义 Guard 时，保持工具原有行为。
        result = ScopeResult(
            decision=ScopeDecision.ALLOW,
            scope="in_scope",
            risk_level="low",
            reason_code="scope_guard_disabled",
            message="allow",
        )
        _record_guard_check(
            state,
            stage="tool",
            layer="hard_rule",
            result=result,
            duration_ms=int((time.perf_counter() - guard_started) * 1000),
        )
        _record_scope_memory(state, result)
        return ToolGuardrailFunctionOutput.allow(
            output_info=result.model_dump(mode="json")
        )

    try:
        result = await state.scope_guard.evaluate(
            state.request_text,
            context_summary=_scope_context_summary(state),
            tool_name=tool_context.tool_name,
            tool_arguments=tool_arguments,
        )
    except ScopeGuardError:
        result = ScopeResult(
            decision=ScopeDecision.CLARIFY,
            scope="ambiguous",
            risk_level="unknown",
            reason_code="scope_guard_unavailable",
            message="暂时无法确认工具调用范围，已阻止图片分析。",
        )
        _record_guard_check(
            state,
            stage="tool",
            layer="semantic",
            result=result,
            duration_ms=int((time.perf_counter() - guard_started) * 1000),
        )
        _record_scope_memory(state, result)
        return ToolGuardrailFunctionOutput.raise_exception(
            output_info=result.model_dump(mode="json")
        )

    _record_guard_check(
        state,
        stage="tool",
        layer="semantic",
        result=result,
        duration_ms=int((time.perf_counter() - guard_started) * 1000),
    )
    _record_scope_memory(state, result)
    output_info = result.model_dump(mode="json")
    if result.decision is ScopeDecision.ALLOW:
        return ToolGuardrailFunctionOutput.allow(output_info=output_info)
    if result.decision is ScopeDecision.CLARIFY:
        return ToolGuardrailFunctionOutput.reject_content(
            result.message,
            output_info=output_info,
        )
    return ToolGuardrailFunctionOutput.raise_exception(output_info=output_info)


@tool_output_guardrail
def generic_output_guard(data: Any) -> ToolGuardrailFunctionOutput:
    """所有 Tool 共用的基础结果检查。"""

    output = data.output
    if output is None:
        return ToolGuardrailFunctionOutput.reject_content(
            "工具没有返回结果。",
            output_info={"reason_code": "empty_tool_output"},
        )
    if isinstance(output, str):
        if not output.strip():
            return ToolGuardrailFunctionOutput.reject_content(
                "工具没有返回有效内容。",
                output_info={"reason_code": "empty_tool_output"},
            )
        if len(output) > 12000:
            return ToolGuardrailFunctionOutput.reject_content(
                "工具返回内容过长，已阻止继续传递。",
                output_info={"reason_code": "tool_output_too_large"},
            )
    else:
        try:
            json.dumps(output, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return ToolGuardrailFunctionOutput.reject_content(
                "工具返回结果不可序列化。",
                output_info={"reason_code": "tool_output_not_serializable"},
            )
    return ToolGuardrailFunctionOutput.allow(
        output_info={"reason_code": "generic_output_valid"}
    )


@tool_output_guardrail
def vision_output_guard(data: Any) -> ToolGuardrailFunctionOutput:
    """图片分析专属检查：拦截未经否定的敏感属性推断。"""

    text = str(data.output or "")
    sensitive_terms = ("真实身份", "身份是", "年龄是", "年龄为", "属于男性", "属于女性")
    safe_refusals = ("无法判断", "不能判断", "不应推断", "无法确认", "不能确认")
    if any(term in text for term in sensitive_terms) and not any(
        refusal in text for refusal in safe_refusals
    ):
        return ToolGuardrailFunctionOutput.reject_content(
            "图片分析结果包含不允许的敏感个人属性推断。",
            output_info={"reason_code": "sensitive_vision_output"},
        )
    return ToolGuardrailFunctionOutput.allow(
        output_info={"reason_code": "vision_output_valid"}
    )


@function_tool(
    is_enabled=vision_tool_enabled,
    needs_approval=make_tool_needs_approval("analyze_current_image"),
    tool_input_guardrails=[photo_tool_scope_guard],
    tool_output_guardrails=[generic_output_guard, vision_output_guard],
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


def _json_safe_metadata(value: Any) -> Any:
    """把 EXIF 中的特殊值转换成可序列化的 JSON 值。"""

    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@tool_output_guardrail
def metadata_output_guard(data: Any) -> ToolGuardrailFunctionOutput:
    """校验元数据工具输出，避免格式错误继续进入 Agent。"""

    output = data.output
    try:
        result = output if isinstance(output, ImageMetadataResult) else ImageMetadataResult.model_validate(output)
    except Exception:
        return ToolGuardrailFunctionOutput.reject_content(
            "图片元数据结果格式无效，无法继续使用。",
            output_info={"reason_code": "invalid_metadata_output"},
        )

    if not result.ok:
        return ToolGuardrailFunctionOutput.reject_content(
            result.error or "图片元数据读取失败。",
            output_info={"reason_code": "metadata_tool_error"},
        )
    if (
        not result.image_id
        or result.width is None
        or result.height is None
        or result.width <= 0
        or result.height <= 0
        or not result.format
    ):
        return ToolGuardrailFunctionOutput.reject_content(
            "图片元数据缺少必要字段，无法继续使用。",
            output_info={"reason_code": "metadata_output_missing_fields"},
        )
    return ToolGuardrailFunctionOutput.allow(
        output_info={"reason_code": "metadata_output_valid"}
    )


@function_tool(
    is_enabled=metadata_tool_enabled,
    needs_approval=make_tool_needs_approval("read_image_metadata"),
    tool_input_guardrails=[photo_tool_scope_guard],
    tool_output_guardrails=[generic_output_guard, metadata_output_guard],
    timeout=30.0,
    timeout_behavior="error_as_result",
)
async def read_image_metadata(
    context: RunContextWrapper[PhotoAgentContext],
    args: ImageMetadataArgs,
) -> ImageMetadataResult:
    """读取图片尺寸、格式和可用 EXIF，不进行视觉推断。"""

    state = context.context
    image_id = args.image_id or state.active_image_id
    if not image_id and len(state.available_image_ids) == 1:
        image_id = state.available_image_ids[0]
    if not image_id:
        return ImageMetadataResult(ok=False, error="当前请求没有可读取元数据的图片。")

    record = state.image_store.get(state.session_id, image_id)
    if record is None:
        return ImageMetadataResult(ok=False, image_id=image_id, error=f"找不到图片引用：{image_id}。")
    if record.ref.startswith(("http://", "https://", "data:")):
        return ImageMetadataResult(ok=False, image_id=image_id, error="当前元数据工具只支持本地图片文件。")
    path = Path(record.ref).expanduser()
    if not path.is_file():
        return ImageMetadataResult(ok=False, image_id=image_id, error=f"图片文件已不存在：{record.ref}")

    state.metadata_done = True
    try:
        with Image.open(path) as image:
            exif_data = {}
            for key, value in image.getexif().items():
                name = ExifTags.TAGS.get(key, str(key))
                exif_data[name] = _json_safe_metadata(value)
            result = {
                "ok": True,
                "image_id": image_id,
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
                "exif": exif_data,
            }
    except (OSError, ValueError) as exc:
        return ImageMetadataResult(ok=False, image_id=image_id, error=f"无法读取图片元数据：{exc}")
    return ImageMetadataResult.model_validate(result)


class AgentRunResult(BaseModel):
    """SDK 一次运行的对外结果，显式带回 session_id。"""

    session_id: str
    final_output: str
    trace_id: str


@dataclass
class PreparedRuntime:
    """一次 Agent 运行所需的 Session、Context 和输入。"""

    session_id: str
    session: Any
    context: PhotoAgentContext
    input_text: str


def _prepare_runtime(
    request: AgentRequest,
    llm_config: LLMConfig,
    db_path: Path,
) -> PreparedRuntime:
    """准备图片引用、Session、Scope Memory 和 SDK Context。"""

    session_id = request.session_id or "default"
    image_store = ImageSessionStore(db_path)
    scope_memory = ScopeMemoryStore(db_path)
    long_term_memory = LongTermMemoryStore(db_path)
    user_id = request.user_id or session_id
    session_summary_record = long_term_memory.get_summary(session_id)
    active_image_id: str | None = None
    available_image_ids: list[str] = []

    if request.images:
        turn_no = image_store.next_turn_no(session_id)
        for image_ref in request.images:
            available_image_ids.append(
                image_store.register(session_id, image_ref, turn_no)
            )
        if len(available_image_ids) == 1:
            active_image_id = available_image_ids[0]
        else:
            active_image_id = image_store.resolve_reference(session_id, request.text)
    else:
        active_image_id = image_store.resolve_reference(session_id, request.text)
        if active_image_id:
            available_image_ids = [
                item.image_id for item in image_store.list(session_id)
            ]

    vision_provider = GLMVisionProvider.from_env() if active_image_id else None
    context = PhotoAgentContext(
        session_id=session_id,
        image_store=image_store,
        request_text=request.text,
        active_image_id=active_image_id,
        available_image_ids=available_image_ids,
        vision_provider=vision_provider,
        scope_guard=SemanticScopeGuard(
            llm_config,
            policy=DEFAULT_SCOPE_POLICY,
            registry=DEFAULT_CAPABILITY_REGISTRY,
        ),
        guard_model=llm_config.model,
        scope_memory=scope_memory,
        scope_policy=DEFAULT_SCOPE_POLICY,
        capability_registry=DEFAULT_CAPABILITY_REGISTRY,
        user_id=user_id,
        long_term_memory=long_term_memory,
        session_summary=(session_summary_record.summary if session_summary_record else None),
        permission_policy=DEFAULT_TOOL_PERMISSION_POLICY,
    )

    input_text = request.text
    memory_context = long_term_memory.context_text(user_id, request.text)
    if memory_context != "暂无用户长期记忆。":
        input_text += (
            "\n系统提供的用户长期记忆（仅在与当前问题相关时使用）："
            f"{memory_context}。"
            "如果用户询问自己的已保存偏好，应直接依据该记忆回答。"
        )
    if session_summary_record is not None:
        input_text += (
            "\n系统提供的本会话摘要："
            f"{session_summary_record.summary}。"
        )
    if input_text.strip() and (request.images or active_image_id):
        available_ids = ", ".join(
            item.image_id for item in image_store.list(session_id)
        )
        input_text += (
            "\n用户已提供或引用图片，如需分析请调用图片工具。"
            f"当前图片：{active_image_id or '未解析'}。"
            f"可用图片引用：{available_ids}。"
        )

    base_session = SQLiteSession(session_id, str(db_path))
    compactor = ContextCompactor()

    def summary_provider() -> str | None:
        latest = long_term_memory.get_summary(session_id)
        return latest.summary if latest else None

    return PreparedRuntime(
        session_id=session_id,
        session=CompactingSession(base_session, compactor, summary_provider),
        context=context,
        input_text=input_text,
    )


def _extract_tool_calls(result: Any, active_image_id: str | None) -> list[dict[str, Any]]:
    """从 SDK 运行结果中提取脱敏后的工具调用摘要。"""

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
    return tool_calls


def _extract_usage(result: Any) -> dict[str, Any]:
    """从 SDK 结果提取模型用量；Provider 不提供时返回空字典。"""

    usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
    if usage is None:
        return {}
    return {
        "requests": getattr(usage, "requests", None),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _append_trace(
    trace_store: JsonlTraceStore,
    *,
    trace_id: str,
    runtime: PreparedRuntime,
    llm_config: LLMConfig,
    retry_tracker: RetryTracker,
    started_at: float,
    status: str,
    error: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> None:
    """统一写入一次运行 Trace，避免成功和异常分支重复构造。"""

    vision_provider = runtime.context.vision_provider
    trace_store.append(
        TraceRecord(
            trace_id=trace_id,
            session_id=runtime.session_id,
            model=llm_config.model,
            vision_model=vision_provider.config.model if vision_provider else None,
            status=status,
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            retry_count=retry_tracker.retry_count,
            failure_type=retry_tracker.failure_type,
            guard_checks=runtime.context.guard_checks,
            tool_calls=tool_calls or [],
            usage=usage or {},
            error=error,
        )
    )


def _persist_memory_after_run(runtime: PreparedRuntime, final_output: str) -> None:
    """保存显式长期记忆和本轮会话摘要。"""

    memory_store = runtime.context.long_term_memory
    if memory_store is None:
        return
    memory_store.remember_explicit(
        runtime.context.user_id,
        runtime.context.request_text,
        runtime.session_id,
    )
    summary = (
        f"用户本轮：{runtime.context.request_text[:240]}；"
        f"Agent 结论：{final_output[:600]}"
    )
    memory_store.upsert_summary(runtime.session_id, summary)


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
            "如果用户询问 EXIF、相机参数、图片尺寸或文件格式，调用 read_image_metadata。"
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
        tools=[analyze_current_image, read_image_metadata],
        input_guardrails=[photo_input_guard],
    )


async def run_photo_agent(request: AgentRequest) -> AgentRunResult:
    """运行一次 Agents SDK Agent。"""

    started_at = time.perf_counter()
    trace_id = uuid.uuid4().hex
    trace_store = JsonlTraceStore()
    retry_tracker = RetryTracker()
    llm_config = LLMConfig.from_env()
    runtime = _prepare_runtime(request, llm_config, Path("data/agent_sessions.db"))
    agent = build_photo_agent(llm_config, retry_tracker)

    try:
        result = await Runner.run(
            agent,
            input=runtime.input_text,
            context=runtime.context,
            session=runtime.session,
            max_turns=4,
        )
    except InputGuardrailTripwireTriggered as exc:
        info = exc.guardrail_result.output.output_info
        final_output = str(info.get("reason", "请求未通过输入护栏。"))
        _append_trace(
            trace_store,
            trace_id=trace_id,
            runtime=runtime,
            llm_config=llm_config,
            retry_tracker=retry_tracker,
            started_at=started_at,
            status="blocked",
            error=final_output,
        )
        return AgentRunResult(session_id=runtime.session_id, final_output=final_output, trace_id=trace_id)
    except ToolInputGuardrailTripwireTriggered as exc:
        info = getattr(getattr(exc, "output", None), "output_info", {}) or {}
        final_output = str(
            info.get("message")
            or info.get("reason")
            or "工具调用未通过执行前护栏。"
        )
        _append_trace(
            trace_store,
            trace_id=trace_id,
            runtime=runtime,
            llm_config=llm_config,
            retry_tracker=retry_tracker,
            started_at=started_at,
            status="blocked",
            error=final_output,
        )
        return AgentRunResult(session_id=runtime.session_id, final_output=final_output, trace_id=trace_id)

    _append_trace(
        trace_store,
        trace_id=trace_id,
        runtime=runtime,
        llm_config=llm_config,
        retry_tracker=retry_tracker,
        started_at=started_at,
        status="success",
        tool_calls=_extract_tool_calls(
            result,
            runtime.context.active_image_id,
        ),
        usage=_extract_usage(result),
    )
    _persist_memory_after_run(runtime, str(result.final_output))
    return AgentRunResult(
        session_id=runtime.session_id,
        final_output=str(result.final_output),
        trace_id=trace_id,
    )
