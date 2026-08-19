"""运行摄影专项三基线评测。

示例：
    PYTHONPATH=. .venv/bin/python eval/run_photo_quality_eval.py \
      --mode photo-coach --split holdout --limit 2

模式：
    text       只调用 DeepSeek，不读取图片
    vision     GLM 观察图片，再由 DeepSeek 根据观察生成回答
    photo-coach 完整 PhotoCoach Agent
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = PROJECT_ROOT / "eval"
CASES_PATH = EVAL_ROOT / "photo_quality_eval.jsonl"
sys.path.insert(0, str(PROJECT_ROOT))

from photo_coach.agents_sdk_app import run_photo_agent
from photo_coach.llm_provider import LLMConfig
from photo_coach.models import AgentRequest
from photo_coach.vision_provider import GLMVisionProvider


def load_cases(path: Path = CASES_PATH, split: str | None = None) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [case for case in cases if split is None or case.get("split") == split]


def resolve_images(images: list[str]) -> list[str]:
    return [str((EVAL_ROOT / image).resolve()) for image in images]


def _text_from_turn(case: dict[str, Any]) -> str:
    if "turns" not in case:
        return case.get("text", "")
    return "\n".join(
        f"第 {index} 轮：{turn.get('text', '')}"
        for index, turn in enumerate(case["turns"], start=1)
    )


def _images_from_case(case: dict[str, Any]) -> list[str]:
    if "turns" not in case:
        return case.get("images", [])
    for turn in reversed(case["turns"]):
        if turn.get("images"):
            return turn["images"]
    return []


async def run_text_baseline(case: dict[str, Any], config: LLMConfig) -> str:
    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )
    response = await client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": "你是摄影助手，只根据用户文字回答摄影问题，不要假设用户提供了图片或 EXIF。",
            },
            {"role": "user", "content": _text_from_turn(case)},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    return response.choices[0].message.content or ""


async def run_vision_baseline(case: dict[str, Any], config: LLMConfig) -> str:
    images = _images_from_case(case)
    provider = GLMVisionProvider.from_env()
    observations = []
    for image in resolve_images(images):
        observations.append(await asyncio.to_thread(provider.analyze, image, _text_from_turn(case)))
    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )
    response = await client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": "你是摄影助手。根据视觉观察事实回答用户，不要编造 EXIF。"},
            {
                "role": "user",
                "content": f"用户问题：{_text_from_turn(case)}\n视觉观察：{'\n\n'.join(observations)}",
            },
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    return response.choices[0].message.content or ""


async def run_photo_coach_baseline(case: dict[str, Any]) -> dict[str, Any]:
    session_id = f"photo-eval-{case['id']}-{int(time.time() * 1000)}"
    final_output = ""
    result = None
    turns = case.get("turns") or [{"text": case.get("text", ""), "images": case.get("images", [])}]
    for turn in turns:
        result = await run_photo_agent(
            AgentRequest(
                text=turn.get("text", ""),
                images=resolve_images(turn.get("images", [])),
                session_id=session_id,
                user_id=f"photo-eval-user-{case['id']}",
            )
        )
        final_output = result.final_output
    return {
        "answer": final_output,
        "trace_id": result.trace_id if result else None,
        "session_id": session_id,
    }


async def evaluate_case(case: dict[str, Any], mode: str, config: LLMConfig) -> dict[str, Any]:
    started = time.perf_counter()
    metadata: dict[str, Any] = {}
    try:
        if mode == "text":
            answer = await run_text_baseline(case, config)
        elif mode == "vision":
            answer = await run_vision_baseline(case, config)
        else:
            result = await run_photo_coach_baseline(case)
            answer = result["answer"]
            metadata.update(result)
        error = None
    except Exception as exc:  # keep the remaining cases running
        answer = ""
        error = f"{type(exc).__name__}: {exc}"
    return {
        "id": case["id"],
        "category": case["category"],
        "mode": mode,
        "answer": answer,
        "answer_nonempty": bool(answer.strip()),
        "expected_tool": case.get("expected_tool"),
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "error": error,
        **metadata,
    }


async def run(mode: str, split: str | None = None, limit: int | None = None) -> dict[str, Any]:
    config = LLMConfig.from_env()
    cases = load_cases(split=split)
    if limit is not None:
        cases = cases[: max(1, limit)]
    results = [await evaluate_case(case, mode, config) for case in cases]
    return {
        "mode": mode,
        "split": split,
        "total": len(results),
        "nonempty_answers": sum(item["answer_nonempty"] for item in results),
        "errors": sum(item["error"] is not None for item in results),
        "avg_latency_ms": sum(item["duration_ms"] for item in results) / len(results) if results else 0.0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("text", "vision", "photo-coach"), required=True)
    parser.add_argument("--split", choices=("dev", "holdout"), default="holdout")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = asyncio.run(run(args.mode, args.split, args.limit))
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
