"""Complete a compact blinded MTRAG review with the OpenAI Responses API."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

MODEL = "gpt-5.4-mini-2026-03-17"
INSTRUCTIONS = """You are a strict, conservative blinded research-data reviewer.
First apply this material-impact test: would replacing one answer with the other
materially change the user's conclusion, action, safety, or understanding of the
core answer? If not, material_difference must be false and the verdict must be
equivalent. Equivalent is the default, including when both answers share a flaw.

Do not prefer an answer merely because it is slightly cleaner, more concise,
longer, more detailed, closer to the expected wording, or contains an optional
fact the other omits. Ignore harmless tangents, minor unsupported details, small
rounding differences, style, and verbosity. If your only reason would be “closer,”
“slightly better,” or “less extra detail,” the verdict must be equivalent.

Set material_difference true and choose answer_a_better or answer_b_better only
for a concrete consequential difference such as a wrong entity or jurisdiction,
a materially false claim, missing the central answer or required action, dangerous
advice, or substantial irrelevance. Use the expected answer as evidence, not as a
wording template. If both candidates answer an apparent question despite a
mismatched expected answer, compare the candidates directly. Choose unclear only
when a fair comparison is genuinely impossible. Never infer which answer used
cache reuse. Give one concise reason naming the material issue, or stating why the
answers are materially equivalent."""


class Judgment(BaseModel):
    material_difference: bool
    verdict: Literal[
        "equivalent", "answer_a_better", "answer_b_better", "unclear"
    ]
    reason: str


def _judge(row: dict, model: str) -> dict:
    evidence = {
        key: row[key]
        for key in (
            "question",
            "expected_answer",
            "answer_a",
            "answer_b",
            "independent_normal_answer",
        )
    }
    response = OpenAI().responses.parse(
        model=model,
        instructions=INSTRUCTIONS,
        input=json.dumps(evidence, ensure_ascii=False),
        text_format=Judgment,
        reasoning={"effort": "low"},
        max_output_tokens=1024,
        store=False,
        prompt_cache_key="mtrag-blinded-semantic-review-v2-conservative",
    )
    parsed = response.output_parsed
    if parsed is None or not parsed.reason.strip():
        raise ValueError("OpenAI response did not contain a complete judgment")
    if parsed.material_difference != (parsed.verdict not in {"equivalent", "unclear"}):
        raise ValueError("OpenAI judgment has inconsistent materiality and verdict")
    return {
        "group_id": row["group_id"],
        "material_difference": parsed.material_difference,
        "verdict": parsed.verdict,
        "reason": parsed.reason.strip(),
        "model": model,
        "response_id": response.id,
        "usage": response.usage.model_dump() if response.usage else None,
    }


def _load_progress(path: Path, group_ids: set[str]) -> dict[str, dict]:
    judgments = {}
    if not path.exists():
        return judgments
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            group_id = row.get("group_id")
            if group_id not in group_ids or group_id in judgments:
                raise ValueError("review progress contains an invalid group ID")
            judgments[group_id] = row
    return judgments


def _apply(compact: dict, judgments: dict[str, dict], model: str) -> None:
    rows = compact["rows"]
    if len(judgments) != len(rows):
        raise ValueError("cannot apply an incomplete OpenAI review")
    for row in rows:
        judgment = judgments[row["group_id"]]
        if judgment.get("model") != model:
            raise ValueError("review progress mixes model snapshots")
        row.update(verdict=judgment["verdict"], reason=judgment["reason"])


def review(result_dir: Path, model: str, workers: int, limit: int | None) -> dict:
    compact_path = result_dir / "manual-review-compact.json"
    progress_path = result_dir / "openai-review-progress.jsonl"
    compact = json.loads(compact_path.read_text(encoding="utf-8"))
    rows = compact["rows"]
    group_ids = {row["group_id"] for row in rows}
    if len(group_ids) != len(rows):
        raise ValueError("compact review contains duplicate group IDs")
    judgments = _load_progress(progress_path, group_ids)
    pending = [row for row in rows if row["group_id"] not in judgments]
    if limit is not None:
        pending = pending[:limit]

    failures = []
    with (
        progress_path.open("a", encoding="utf-8") as sink,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        futures = {executor.submit(_judge, row, model): row for row in pending}
        for completed, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                judgment = future.result()
            except Exception as error:  # noqa: BLE001
                failures.append(f"{row['group_id']}: {error}")
                continue
            judgments[judgment["group_id"]] = judgment
            sink.write(json.dumps(judgment, ensure_ascii=False) + "\n")
            sink.flush()
            if completed % 25 == 0 or completed == len(pending):
                print(f"Reviewed {len(judgments)}/{len(rows)} groups", flush=True)

    if len(judgments) == len(rows):
        _apply(compact, judgments, model)
        compact_path.write_text(
            json.dumps(compact, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return {
        "completed": len(judgments),
        "total": len(rows),
        "failures": failures,
        "applied": len(judgments) == len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = review(args.result_dir, args.model, args.workers, args.limit)
    print(json.dumps(result, indent=2))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
