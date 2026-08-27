"""Freeze and resolve ChatRAG external policy-evaluation cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmarks.block_dataset import DatasetSplit
from benchmarks.chatrag import (
    CHATRAG_SPLIT_SEED,
    chatrag_conversation_split,
)
from benchmarks.mtrag import MtragTask, mtrag_source_sha256
from benchmarks.mtrag_pilot import mtrag_testable_blocks
from benchmarks.mtrag_trace import build_mtrag_transition_trace
from benchmarks.reviewed_mtrag_counterfactual import QWEN3_REVIEWED_REFERENCE_GATE
from benchmarks.schema import WorkloadTrace


@dataclass(frozen=True)
class ChatRagPolicyCase:
    trace: WorkloadTrace
    split: DatasetSplit
    collection: str


def freeze_chatrag_policy_manifest(
    coverage: Mapping[str, object],
    *,
    source_sha256: str,
    max_model_len: int = 8192,
) -> dict[str, object]:
    """Keep every executable transition and freeze its conversation split."""
    block_size = coverage.get("block_size")
    rows = coverage.get("transitions")
    if (
        coverage.get("analysis") != "chatrag-natural-block-coverage"
        or isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or not isinstance(rows, list)
        or not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or max_model_len < 1
    ):
        raise ValueError("input is not a valid ChatRAG coverage artifact")
    selected = []
    counts: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("coverage transition must be an object")
        _, testable, _ = mtrag_testable_blocks(
            row,
            block_size=block_size,
            include_repacking=True,
        )
        opportunity = row.get("reuse_opportunity")
        prompt_tokens = (
            opportunity.get("current_token_count")
            if isinstance(opportunity, Mapping)
            else None
        )
        if not testable or not isinstance(prompt_tokens, int) or prompt_tokens > max_model_len:
            continue
        identity = {
            field: row.get(field)
            for field in (
                "conversation_id",
                "collection",
                "previous_task_id",
                "current_task_id",
            )
        }
        if any(not isinstance(value, str) or not value for value in identity.values()):
            raise ValueError("coverage transition has invalid identity")
        split = chatrag_conversation_split(identity["conversation_id"]).value
        selected.append(
            {
                **identity,
                "split": split,
                "current_prompt_tokens": prompt_tokens,
                "testable_block_count": len(testable),
            }
        )
        counts[split] += 1
    return {
        "schema_version": 1,
        "selection": "chatrag-external-policy-evaluation",
        "source_dataset_sha256": source_sha256,
        "source_model": coverage.get("model"),
        "subset": coverage.get("subset"),
        "max_contexts": coverage.get("max_contexts"),
        "block_size": block_size,
        "max_model_len": max_model_len,
        "split_seed": CHATRAG_SPLIT_SEED,
        "transition_count": len(selected),
        "split_transition_counts": dict(counts),
        "transitions": selected,
    }


def build_chatrag_policy_cases(
    tasks: Sequence[MtragTask],
    manifest: Mapping[str, object],
    *,
    split: DatasetSplit,
    expected_model: str,
) -> tuple[ChatRagPolicyCase, ...]:
    if (
        manifest.get("selection") != "chatrag-external-policy-evaluation"
        or manifest.get("split_seed") != CHATRAG_SPLIT_SEED
        or manifest.get("source_model") != expected_model
        or not isinstance(manifest.get("transitions"), list)
    ):
        raise ValueError("ChatRAG policy manifest is incompatible")
    tasks_by_id = {task.task_id: task for task in tasks}
    cases = []
    for row in manifest["transitions"]:
        if not isinstance(row, Mapping) or row.get("split") != split.value:
            continue
        try:
            previous = tasks_by_id[row["previous_task_id"]]
            current = tasks_by_id[row["current_task_id"]]
        except (KeyError, TypeError) as error:
            raise ValueError("ChatRAG policy task is missing") from error
        if (
            row.get("conversation_id") != current.conversation_id
            or row.get("collection") != current.collection
            or chatrag_conversation_split(current.conversation_id) is not split
        ):
            raise ValueError("ChatRAG policy identity or split changed")
        cases.append(
            ChatRagPolicyCase(
                trace=build_mtrag_transition_trace(
                    previous,
                    current,
                    quality_gate=QWEN3_REVIEWED_REFERENCE_GATE,
                    workload="chatrag_rag",
                    source_name="NVIDIA ChatRAG-Bench",
                    transition_prefix="chatrag",
                ),
                split=split,
                collection=current.collection,
            )
        )
    if not cases:
        raise ValueError(f"ChatRAG manifest has no {split.value} cases")
    return tuple(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    manifest = freeze_chatrag_policy_manifest(
        coverage,
        source_sha256=mtrag_source_sha256(args.source_dataset),
        max_model_len=args.max_model_len,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Frozen {manifest['transition_count']} ChatRAG transitions at "
        f"{manifest['split_transition_counts']}"
    )


if __name__ == "__main__":
    main()
