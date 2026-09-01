"""Create runtime-model copies for frozen validation reuse budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.train_logistic_selector import REUSE_BUDGETS


def export_reuse_budget_models(
    model_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    payload: dict[str, Any] = json.loads(model_path.read_text(encoding="utf-8"))
    thresholds = payload.get("reuse_budget_thresholds")
    expected = {f"{budget:.2f}" for budget in REUSE_BUDGETS}
    if payload.get("schema_version") != 1 or set(thresholds or ()) != expected:
        raise ValueError("runtime model has no complete reuse-budget thresholds")
    source_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for target, threshold in thresholds.items():
        variant = dict(payload)
        variant["selected_repair_threshold"] = float(threshold)
        variant["reuse_budget"] = {
            "target_reuse_rate": float(target),
            "source_model_sha256": source_sha256,
        }
        suffix = f"{round(float(target) * 100):03d}"
        output = output_dir / f"{model_path.stem}-reuse-{suffix}.json"
        output.write_text(json.dumps(variant, indent=2) + "\n", encoding="utf-8")
        outputs[target] = output
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = export_reuse_budget_models(args.model, args.output_dir)
    for target, output in outputs.items():
        print(f"reuse={target} model={output}")


if __name__ == "__main__":
    main()
