"""校验摄影专项评测集结构和样例图片引用。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {"id", "split", "category", "expected_decision", "rubric"}


def load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate(cases: list[dict], image_root: Path | None = None) -> dict:
    errors: list[str] = []
    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)):
        errors.append("存在重复 id")
    for index, case in enumerate(cases, start=1):
        missing = REQUIRED - set(case)
        if missing:
            errors.append(f"第 {index} 条缺少字段：{sorted(missing)}")
        has_turns = isinstance(case.get("turns"), list) and bool(case["turns"])
        if has_turns:
            turns = case["turns"]
            for turn in turns:
                if "text" not in turn or "images" not in turn:
                    errors.append(f"{case.get('id')} turn 缺少 text 或 images")
        else:
            if "text" not in case or "images" not in case:
                errors.append(f"{case.get('id')} 缺少 text 或 images")
        if case.get("split") not in {"dev", "holdout"}:
            errors.append(f"{case.get('id')} split 必须是 dev 或 holdout")
        if case.get("expected_decision") not in {"allow", "refuse", "clarify"}:
            errors.append(f"{case.get('id')} expected_decision 无效")
        if case.get("expected_decision") == "allow":
            for key in ("expected_capability", "expected_tool"):
                if key not in case:
                    errors.append(f"{case.get('id')} 缺少 {key}")
        image_values = list(case.get("images", []))
        for turn in case.get("turns", []):
            image_values.extend(turn.get("images", []))
        if image_root:
            for image in image_values:
                if not (image_root / image).is_file():
                    errors.append(f"{case.get('id')} 找不到图片：{image}")
    return {
        "total": len(cases),
        "dev": sum(case.get("split") == "dev" for case in cases),
        "holdout": sum(case.get("split") == "holdout" for case in cases),
        "categories": sorted({case.get("category") for case in cases}),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/photo_quality_eval.jsonl")
    parser.add_argument("--image-root", default=None)
    args = parser.parse_args()
    report = validate(
        load_cases(Path(args.dataset)),
        Path(args.image_root) if args.image_root else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
