"""运行不依赖模型调用的多轮图片会话评测。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from photo_coach.image_session_store import ImageSessionStore


CASES_PATH = PROJECT_ROOT / "eval" / "multiturn_cases.jsonl"


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_case(case: dict, root: Path) -> dict:
    db_path = root / f"{case['id']}.db"
    session_id = case["session_id"]
    store = ImageSessionStore(db_path)

    for index, turn in enumerate(case["turns"], start=1):
        image_ref = turn.get("image")
        if image_ref:
            turn_no = store.next_turn_no(session_id)
            store.register(session_id, image_ref, turn_no)

        if index < len(case["turns"]):
            continue

        if case.get("reopen_before_final"):
            store = ImageSessionStore(db_path)

        resolved = store.resolve_reference(session_id, turn.get("text", ""))
        expected = case["expected"]
        record = store.get(session_id, resolved) if resolved else None
        actual = {
            "resolved_image_id": resolved,
            "reuse_image": resolved is not None,
            "should_clarify": resolved is None,
        }
        if "ref_exists" in expected:
            actual["ref_exists"] = bool(record and Path(record.ref).is_file())

        passed = all(actual.get(key) == value for key, value in expected.items())
        return {
            "id": case["id"],
            "passed": passed,
            "expected": expected,
            "actual": actual,
        }

    raise ValueError(f"case has no turns: {case['id']}")


def main() -> None:
    cases = load_cases()
    with tempfile.TemporaryDirectory() as directory:
        results = [evaluate_case(case, Path(directory)) for case in cases]

    passed = sum(result["passed"] for result in results)
    print(json.dumps({
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
        "results": results,
    }, ensure_ascii=False, indent=2))

    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
