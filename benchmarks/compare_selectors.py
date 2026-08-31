"""Compare candidate-block selectors on validation or the held-out test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from benchmarks.train_boosted_selector import boosted_model, train_boosted_selector
from benchmarks.train_logistic_selector import (
    baseline_metrics,
    evaluate_selector,
    load_selector_dataset,
    logistic_model,
    operating_points,
    select_repair_threshold,
    train_logistic_selector,
)
from benchmarks.train_mlp_selector import (
    _mlp_pipeline,
    _training_groups,
    balanced_binary_training_rows,
    train_mlp_selector,
)
from cacheselect.selector_features import (
    BASELINE_FEATURE_SCHEMA,
    FEATURE_SCHEMAS,
    SelectorFeatureSchema,
    selector_feature_schema,
)


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


def compare_cross_validated_selectors(
    dataset_path: Path,
    *,
    minimum_repair_recall: float = 0.95,
    feature_schema: SelectorFeatureSchema = BASELINE_FEATURE_SCHEMA,
    folds: int = 5,
) -> dict[str, object]:
    """Compare selectors using out-of-fold predictions from unseen conversations."""
    features, labels = load_selector_dataset(
        dataset_path,
        feature_schema=feature_schema,
        require_validation=False,
    )["train"]
    groups = _training_groups(dataset_path)
    if len(groups) != len(labels):
        raise ValueError("training groups and examples are misaligned")
    models = {}
    for name, factory, balance in (
        ("logistic_regression", logistic_model, False),
        ("hist_gradient_boosting", boosted_model, False),
        ("mlp", lambda: _mlp_pipeline((8,)), True),
    ):
        probabilities = np.empty(len(labels))
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=0
        )
        for train, held_out in splitter.split(features, labels, groups):
            training = (features[train], labels[train])
            if balance:
                training = balanced_binary_training_rows(*training)
            model = factory()
            model.fit(*training)
            probabilities[held_out] = model.predict_proba(features[held_out])[:, 1]
        threshold = select_repair_threshold(
            labels,
            probabilities,
            minimum_repair_recall=minimum_repair_recall,
        )
        models[name] = {
            "selected_threshold": threshold,
            "metrics": evaluate_selector(
                labels,
                probabilities >= threshold,
                repair_probabilities=probabilities,
            ),
            "operating_points": operating_points(labels, probabilities),
        }
    return {
        "dataset": str(dataset_path),
        "evaluation_split": "conversation_grouped_cross_validation",
        "folds": folds,
        "conversation_groups": len(set(groups)),
        "test_split_evaluated": False,
        "minimum_repair_recall": minimum_repair_recall,
        "feature_schema": feature_schema.name,
        "feature_names": list(feature_schema.feature_names),
        "models": models,
        "baselines": baseline_metrics(
            features, labels, feature_schema=feature_schema
        ),
    }


def _clustered_intervals(
    labels: np.ndarray,
    predicted_repair: np.ndarray,
    groups: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Bootstrap test metrics by conversation rather than by correlated block."""
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2 or samples < 100:
        raise ValueError("clustered bootstrap requires two groups and 100 samples")
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    generator = np.random.default_rng(seed)
    values = {
        name: []
        for name in (
            "repair_recall",
            "safe_reuse_precision",
            "selected_reuse_rate",
        )
    }
    while len(values["repair_recall"]) < samples:
        selected = generator.choice(unique_groups, len(unique_groups), replace=True)
        sampled = np.concatenate([indices[group] for group in selected])
        if not np.any(labels[sampled]):
            continue
        metrics = evaluate_selector(labels[sampled], predicted_repair[sampled])
        for name, metric_values in values.items():
            metric_values.append(float(metrics[name]))
    return {
        name: np.quantile(metric_values, (0.025, 0.975)).tolist()
        for name, metric_values in values.items()
    }


def compare_heldout_selectors(
    dataset_path: Path,
    *,
    selected_model: str,
    minimum_repair_recall: float = 0.95,
    feature_schema: SelectorFeatureSchema = BASELINE_FEATURE_SCHEMA,
    bootstrap_samples: int = 10_000,
) -> dict[str, object]:
    """Evaluate frozen validation thresholds once on conversation-grouped test data."""
    test_features, test_labels = load_selector_dataset(
        dataset_path, feature_schema=feature_schema
    )["test"]
    with dataset_path.open(newline="", encoding="utf-8") as input_file:
        test_rows = [
            row for row in csv.DictReader(input_file) if row.get("split") == "test"
        ]
    if len(test_rows) != len(test_labels):
        raise ValueError("test rows and feature rows are misaligned")
    groups = np.asarray([row.get("mtrag_conversation_id", "") for row in test_rows])
    if not np.all(groups):
        raise ValueError("test rows require mtrag_conversation_id")

    trainers = (
        ("logistic_regression", train_logistic_selector),
        ("hist_gradient_boosting", train_boosted_selector),
        ("mlp", train_mlp_selector),
    )
    if selected_model not in {name for name, _ in trainers}:
        raise ValueError(f"unknown selected model: {selected_model}")
    models = {}
    for model_name, trainer in trainers:
        model, training = trainer(
            dataset_path,
            minimum_repair_recall=minimum_repair_recall,
            feature_schema=feature_schema,
        )
        threshold = float(training["selected_threshold"])
        probabilities = model.predict_proba(test_features)[:, 1]
        predicted_repair = probabilities >= threshold
        missed = np.flatnonzero((test_labels == 1) & ~predicted_repair)
        models[model_name] = {
            "frozen_validation_threshold": threshold,
            "metrics": evaluate_selector(
                test_labels,
                predicted_repair,
                repair_probabilities=probabilities,
            ),
            "conversation_bootstrap_95pct_intervals": _clustered_intervals(
                test_labels,
                predicted_repair,
                groups,
                samples=bootstrap_samples,
            ),
            "missed_repairs": [
                {
                    "trace_id": test_rows[index].get("trace_id"),
                    "trial_id": test_rows[index].get("trial_id"),
                    "candidate_block_index": int(
                        test_rows[index]["candidate_block_index"]
                    ),
                    "repair_probability": float(probabilities[index]),
                    "label_reason": test_rows[index].get("label_reason"),
                    "review_reason": test_rows[index].get("review_reason"),
                }
                for index in missed
            ],
        }
    return {
        "schema_version": 1,
        "analysis": "held-out-selector-comparison",
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "evaluation_split": "test",
        "test_split_used_for_training_or_selection": False,
        "selected_model_before_test": selected_model,
        "minimum_validation_repair_recall": minimum_repair_recall,
        "feature_schema": feature_schema.name,
        "test_examples": len(test_labels),
        "test_conversations": len(set(groups)),
        "bootstrap_samples": bootstrap_samples,
        "models": models,
    }


# Run the comparison and save one auditable JSON validation report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-repair-recall", type=float, default=0.95)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--selected-model",
        choices=("logistic_regression", "hist_gradient_boosting", "mlp"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--feature-schema",
        choices=sorted(FEATURE_SCHEMAS),
        default=BASELINE_FEATURE_SCHEMA.name,
    )
    args = parser.parse_args()

    if args.evaluate_test and args.cross_validate:
        parser.error("choose either --evaluate-test or --cross-validate")
    if args.evaluate_test:
        if args.selected_model is None:
            parser.error("--evaluate-test requires --selected-model")
        report = compare_heldout_selectors(
            args.dataset,
            selected_model=args.selected_model,
            minimum_repair_recall=args.minimum_repair_recall,
            feature_schema=selector_feature_schema(args.feature_schema),
            bootstrap_samples=args.bootstrap_samples,
        )
    elif args.cross_validate:
        report = compare_cross_validated_selectors(
            args.dataset,
            minimum_repair_recall=args.minimum_repair_recall,
            feature_schema=selector_feature_schema(args.feature_schema),
            folds=args.folds,
        )
    else:
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
    print(f"Saved {report['evaluation_split']} comparison to {args.output}")


if __name__ == "__main__":
    main()
