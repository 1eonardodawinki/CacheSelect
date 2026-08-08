import json
from copy import deepcopy

import pytest

from benchmarks.analyze_repair_policy_shadow import (
    CONDITIONS,
    analyze_repair_policy_shadow,
)


# Build one synthetic four-request result with the requested repair metrics.
def make_result(metrics: dict) -> dict:
    server_metrics = {
        "cacheselect_repair_selector": metrics["selector"],
        "cacheselect_candidate_tokens": metrics["candidate_tokens"],
        "cacheselect_repair_tokens": metrics["repair_tokens"],
        "cacheselect_skipped_repair_tokens": metrics[
            "skipped_repair_tokens"
        ],
    }
    observations = []
    for index in range(4):
        observations.append(
            {
                "request_id": f"request-{index}",
                "prompt_token_ids": [index, index + 1],
                "cached_tokens": index * 16,
                "output_text": f"answer-{index}",
                "quality": {"passed": True},
                "server_metrics": server_metrics if index == 2 else {},
            }
        )
    return {
        "observations": observations,
        "request_ledger_summary": {
            "started": 4,
            "completed": 4,
            "failed": 0,
        },
    }


# Write a complete synthetic matrix into the analyzer's expected file layout.
def write_matrix(tmp_path) -> None:
    for condition, metrics in CONDITIONS.items():
        path = tmp_path / f"rag-{condition}.json"
        path.write_text(json.dumps(make_result(metrics)))


# Check that a complete policy matrix produces the expected compact summary.
def test_analyze_repair_policy_shadow(tmp_path) -> None:
    write_matrix(tmp_path)

    summary = analyze_repair_policy_shadow(tmp_path)

    assert summary["conditions"] == CONDITIONS
    assert summary["behavior_identical"]
    assert summary["quality_passed"]


# Check that the analyzer detects a shadow policy changing model output.
def test_analyze_repair_policy_shadow_rejects_behavior_change(tmp_path) -> None:
    write_matrix(tmp_path)
    path = tmp_path / "rag-edit-radius-1.json"
    changed = json.loads(path.read_text())
    changed["observations"][0]["output_text"] = "different answer"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="changed request behavior"):
        analyze_repair_policy_shadow(tmp_path)


# Check that the analyzer rejects incorrect selector counts.
def test_analyze_repair_policy_shadow_rejects_wrong_counts(tmp_path) -> None:
    write_matrix(tmp_path)
    path = tmp_path / "rag-edit-radius-1.json"
    changed = json.loads(path.read_text())
    changed_metrics = deepcopy(changed["observations"][2]["server_metrics"])
    changed_metrics["cacheselect_repair_tokens"] = 64
    changed["observations"][2]["server_metrics"] = changed_metrics
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="expected repair metrics"):
        analyze_repair_policy_shadow(tmp_path)
