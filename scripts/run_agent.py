"""运行一次基于 OpenAI Agents SDK 的 PhotoCoach。"""

from __future__ import annotations

import argparse
import asyncio

from photo_coach.agents_sdk_app import run_photo_agent
from photo_coach.models import AgentRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="用户的摄影问题")
    parser.add_argument("--image", default=None, help="可选图片路径")
    parser.add_argument("--session-id", default="default", help="会话 ID")
    parser.add_argument("--user-id", default=None, help="可选用户 ID，用于跨会话长期记忆")
    args = parser.parse_args()

    request = AgentRequest(
        text=args.text,
        images=[args.image] if args.image else [],
        session_id=args.session_id,
        user_id=args.user_id,
    )
    result = asyncio.run(run_photo_agent(request))
    print(f"session_id={result.session_id}")
    print(f"trace_id={result.trace_id}")
    print(result.final_output)


if __name__ == "__main__":
    main()
