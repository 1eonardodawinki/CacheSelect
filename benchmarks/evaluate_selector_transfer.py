"""Evaluate synthetic-trained selectors on an external causal dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.train_boosted_selector import train_boosted_selector
from benchmarks.train_logistic_selector import (
    baseline_metrics,
    evaluate_selector,
    train_logistic_selector,
)
from benchmarks.train_mlp_selector import train_mlp_selector
from cacheselect.selector_features import (
    CONTEXT_FEATURE_SCHEMA,
    FEATURE_SCHEMAS,
    SelectorFeatureSchema,
    selector_feature_schema,
)


# Parse one external table without requiring synthetic split membership.
def load_external_selector_rows(
    path: Path,
    *,
    feature_schema: SelectorFeatureSchema,
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows = []
    labels = []
    with path.open(newline="", encoding="utf-8") as input_file:
        for line_number, row in enumerate(csv.DictReader(input_file), start=2):
            decision = row.get("decision")
            if decision not in {"repair", "reuse"}:
                raise ValueError(f"line {line_number}: invalid decision {decision!r}")
            try:
                values = []
                for name in feature_schema.feature_names:
                    value = row[name]
                    if name in feature_schema.boolean_features:
                        if value not in {"True", "False"}:
                            raise ValueError(f"invalid boolean {value!r}")
                        values.append(float(value == "True"))
                    else:
                        values.append(float(value))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"line {line_number}: invalid selector feature"
                ) from error
            feature_rows.append(values)
            labels.append(int(decision == "repair"))
    if not feature_rows:
        raise ValueError("external selector dataset is empty")
    return (
        np.asarray(feature_rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
    )


# Train only on synthetic data and evaluate frozen decisions on natural rows.
def evaluate_selector_transfer(
    source_dataset: Path,
    target_dataset: Path,
    *,
    feature_schema: SelectorFeatureSchema = CONTEXT_FEATURE_SCHEMA,
    minimum_repair_recall: float = 0.95,
) -> dict[str, object]:
    target_features, target_labels = load_external_selector_rows(
        target_dataset,
        feature_schema=feature_schema,
    )
    trainers = (
        ("logistic_regression", train_logistic_selector),
        ("hist_gradient_boosting", train_boosted_selector),
        ("mlp", train_mlp_selector),
    )
    models = {}
    for model_name, trainer in trainers:
        model, source_report = trainer(
            source_dataset,
            minimum_repair_recall=minimum_repair_recall,
            feature_schema=feature_schema,
        )
        threshold = float(source_report["selected_threshold"])
        probabilities = model.predict_proba(target_features)[:, 1]
        models[model_name] = {
            "source_validation": source_report["validation"][model_name],
            "frozen_threshold": threshold,
            "target": evaluate_selector(
                target_labels,
                probabilities >= threshold,
                repair_probabilities=probabilities,
            ),
        }
    return {
        "schema_version": 1,
        "analysis": "synthetic-to-natural-selector-transfer",
        "feature_schema": feature_schema.name,
        "feature_names": list(feature_schema.feature_names),
        "source_dataset": str(source_dataset),
        "source_sha256": hashlib.sha256(source_dataset.read_bytes()).hexdigest(),
        "target_dataset": str(target_dataset),
        "target_sha256": hashlib.sha256(target_dataset.read_bytes()).hexdigest(),
        "minimum_source_validation_repair_recall": minimum_repair_recall,
        "synthetic_test_split_evaluated": False,
        "target_used_for_training": False,
        "target_examples": len(target_labels),
        "target_repair_labels": int(np.sum(target_labels)),
        "target_reuse_labels": int(np.sum(target_labels == 0)),
        "target_baselines": baseline_metrics(
            target_features,
            target_labels,
            feature_schema=feature_schema,
        ),
        "models": models,
    }


# Run the transfer experiment and save one provenance-bound JSON report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-schema",
        choices=sorted(FEATURE_SCHEMAS),
        default=CONTEXT_FEATURE_SCHEMA.name,
    )
    parser.add_argument("--minimum-repair-recall", type=float, default=0.95)
    args = parser.parse_args()
    report = evaluate_selector_transfer(
        args.source_dataset,
        args.target_dataset,
        feature_schema=selector_feature_schema(args.feature_schema),
        minimum_repair_recall=args.minimum_repair_recall,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for model_name, result in report["models"].items():
        metrics = result["target"]
        print(
            f"{model_name}: repair_recall={metrics['repair_recall']:.3f} "
            f"selected_reuse_rate={metrics['selected_reuse_rate']:.3f}"
        )
    print(f"Saved transfer report to {args.output}")


if __name__ == "__main__":
    main()
