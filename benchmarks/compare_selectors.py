"""Compare candidate-block selectors without opening the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cacheselect.selector_features import (
    BASELINE_FEATURE_SCHEMA,
    FEATURE_SCHEMAS,
    SelectorFeatureSchema,
    selector_feature_schema,
)
from benchmarks.train_boosted_selector import train_boosted_selector
from benchmarks.train_logistic_selector import train_logistic_selector
from benchmarks.train_mlp_selector import train_mlp_selector


# Train all learned selectors and place their validation results side by side.
def compare_validation_selectors(
    dataset_path: Path,
    *,
    minimum_repair_recall: float = 0.95,
    feature_schema: SelectorFeatureSchema = BASELINE_FEATURE_SCHEMA,
) -> dict[str, object]:
    _, logistic = train_logistic_selector(
        dataset_path,
        minimum_repair_recall=minimum_repair_recall,
        feature_schema=feature_schema,
    )
    _, boosted = train_boosted_selector(
        dataset_path,
        minimum_repair_recall=minimum_repair_recall,
        feature_schema=feature_schema,
    )
    _, mlp = train_mlp_selector(
        dataset_path,
        minimum_repair_recall=minimum_repair_recall,
        feature_schema=feature_schema,
    )
    logistic_validation = logistic["validation"]
    boosted_validation = boosted["validation"]
    mlp_validation = mlp["validation"]
    return {
        "dataset": str(dataset_path),
        "evaluation_split": "validation",
        "test_split_evaluated": False,
        "minimum_repair_recall": minimum_repair_recall,
        "feature_schema": feature_schema.name,
        "feature_names": logistic["feature_names"],
        "models": {
            "logistic_regression": {
                "selected_threshold": logistic["selected_threshold"],
                "metrics": logistic_validation["logistic_regression"],
                "operating_points": logistic_validation["operating_points"],
            },
            "hist_gradient_boosting": {
                "selected_threshold": boosted["selected_threshold"],
                "hyperparameters": boosted["hyperparameters"],
                "metrics": boosted_validation["hist_gradient_boosting"],
                "operating_points": boosted_validation["operating_points"],
            },
            "mlp": {
                "selected_threshold": mlp["selected_threshold"],
                "hyperparameters": mlp["hyperparameters"],
                "training": mlp["training"],
                "metrics": mlp_validation["mlp"],
                "operating_points": mlp_validation["operating_points"],
            },
        },
        "baselines": logistic_validation["baselines"],
    }


# Run the comparison and save one auditable JSON validation report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-repair-recall", type=float, default=0.95)
    parser.add_argument(
        "--feature-schema",
        choices=sorted(FEATURE_SCHEMAS),
        default=BASELINE_FEATURE_SCHEMA.name,
    )
    args = parser.parse_args()

    report = compare_validation_selectors(
        args.dataset,
        minimum_repair_recall=args.minimum_repair_recall,
        feature_schema=selector_feature_schema(args.feature_schema),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for model_name, result in report["models"].items():
        metrics = result["metrics"]
        print(
            f"{model_name}: repair_recall={metrics['repair_recall']:.3f} "
            f"selected_reuse_rate={metrics['selected_reuse_rate']:.3f}"
        )
    print(f"Saved validation comparison to {args.output}")


if __name__ == "__main__":
    main()
