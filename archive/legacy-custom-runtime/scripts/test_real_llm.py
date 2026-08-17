"""Smoke test for the real OpenAI-compatible task-understanding provider."""

from __future__ import annotations

import argparse
import json

from photo_coach.llm_provider import OpenAICompatibleJSONProvider
from photo_coach.llm_router import LLMTaskRouter
from photo_coach.models import AgentRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="这张照片为什么人物脸部很暗？")
    parser.add_argument("--image", default="portrait.jpg")
    parser.add_argument(
        "--env-file",
        default=None,
        help="可选 .env 路径；也可以使用 PHOTOCOACH_ENV_FILE",
    )
    args = parser.parse_args()

    provider = OpenAICompatibleJSONProvider.from_env(args.env_file)
    router = LLMTaskRouter(provider)
    result = router.route(
        AgentRequest(text=args.text, images=[args.image] if args.image else [])
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
