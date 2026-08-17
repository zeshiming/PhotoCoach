"""Agents SDK 模型重试策略和可观测状态。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class RetryTracker:
    """记录一次 Agent 运行中发生的模型重试。"""

    retry_count: int = 0
    failure_type: str | None = None


def classify_retry_context(context: Any) -> str:
    """把 Agents SDK RetryPolicyContext 转成可读的失败类型。"""

    normalized = getattr(context, "normalized", None)
    if getattr(normalized, "is_timeout", False):
        return "timeout"
    if getattr(normalized, "is_network_error", False):
        return "network_error"
    status_code = getattr(normalized, "status_code", None)
    if status_code is not None:
        return f"http_{status_code}"
    return type(getattr(context, "error", None)).__name__


def build_retry_policy(tracker: RetryTracker):
    """创建只允许瞬时错误重试的 SDK RetryPolicy。"""

    def policy(context: Any) -> bool:
        tracker.retry_count = max(tracker.retry_count, int(context.attempt))
        tracker.failure_type = classify_retry_context(context)
        normalized = getattr(context, "normalized", None)
        if getattr(normalized, "is_network_error", False):
            return True
        if getattr(normalized, "is_timeout", False):
            return True
        return getattr(normalized, "status_code", None) in RETRYABLE_STATUS_CODES

    return policy
