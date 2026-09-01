"""Train and evaluate a logistic candidate-block repair selector."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cacheselect.selector_features import (
    BASELINE_FEATURE_SCHEMA,
    FEATURE_SCHEMAS,
    SelectorFeatureSchema,
    selector_feature_schema,
)

SPLITS = ("train", "validation", "test")
SAFETY_RECALL_TARGETS = (0.90, 0.95, 0.99, 1.0)
REUSE_BUDGETS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.0)


def logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=0,
                ),
            ),
        ]
    )


# Parse one serialized boolean without accepting arbitrary truthy strings.
def _parse_boolean(value: str, *, field_name: str) -> float:
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    raise ValueError(f"invalid boolean for {field_name}: {value!r}")


# Read only runtime-safe features and labels from the flat block dataset.
def load_selector_dataset(
    path: Path,
    *,
    feature_schema: SelectorFeatureSchema = BASELINE_FEATURE_SCHEMA,
    require_validation: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[list[float], int]]] = {split: [] for split in SPLITS}
    with path.open(newline="") as input_file:
        for line_number, row in enumerate(csv.DictReader(input_file), start=2):
            split = row.get("split")
            if split not in grouped:
                raise ValueError(f"line {line_number}: invalid split {split!r}")
            decision = row.get("decision")
            if decision not in ("repair", "reuse"):
                raise ValueError(f"line {line_number}: invalid decision {decision!r}")
            try:
                features = [
                    (
                        _parse_boolean(row[name], field_name=name)
                        if name in feature_schema.boolean_features
                        else float(row[name])
                    )
                    for name in feature_schema.feature_names
                ]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"line {line_number}: invalid selector feature"
                ) from error
            grouped[split].append((features, int(decision == "repair")))

    dataset = {}
    for split, examples in grouped.items():
        if not examples:
            if split == "test" or (split == "validation" and not require_validation):
                continue
            raise ValueError(f"dataset split {split!r} is empty")
        feature_rows, labels = zip(*examples, strict=True)
        dataset[split] = (
            np.asarray(feature_rows, dtype=np.float64),
            np.asarray(labels, dtype=np.int64),
        )
    return dataset


# Measure safety and reuse opportunity for one vector of binary decisions.
def evaluate_selector(
    labels: np.ndarray,
    predicted_repair: np.ndarray,
    *,
    repair_probabilities: np.ndarray | None = None,
) -> dict[str, float | int]:
    labels = labels.astype(bool)
    predicted_repair = predicted_repair.astype(bool)
    true_repair = int(np.sum(labels & predicted_repair))
    missed_repair = int(np.sum(labels & ~predicted_repair))
    unnecessary_repair = int(np.sum(~labels & predicted_repair))
    safe_reuse = int(np.sum(~labels & ~predicted_repair))
    repair_precision = true_repair / max(true_repair + unnecessary_repair, 1)
    repair_recall = true_repair / max(true_repair + missed_repair, 1)
    metrics: dict[str, float | int] = {
        "examples": len(labels),
        "true_repair": true_repair,
        "missed_repair": missed_repair,
        "unnecessary_repair": unnecessary_repair,
        "safe_reuse": safe_reuse,
        "accuracy": (true_repair + safe_reuse) / len(labels),
        "repair_precision": repair_precision,
        "repair_recall": repair_recall,
        "repair_f1": (
            2
            * repair_precision
            * repair_recall
            / max(repair_precision + repair_recall, np.finfo(float).eps)
        ),
        "selected_reuse_rate": (safe_reuse + missed_repair) / len(labels),
        "safe_reuse_precision": safe_reuse / max(safe_reuse + missed_repair, 1),
    }
    if repair_probabilities is not None:
        metrics["roc_auc"] = float(roc_auc_score(labels, repair_probabilities))
        metrics["average_precision"] = float(
            average_precision_score(labels, repair_probabilities)
        )
    return metrics


# Choose the most permissive threshold that satisfies validation repair recall.
def select_repair_threshold(
    labels: np.ndarray,
    repair_probabilities: np.ndarray,
    *,
    minimum_repair_recall: float,
) -> float:
    if not 0 < minimum_repair_recall <= 1:
        raise ValueError("minimum_repair_recall must be in (0, 1]")
    thresholds = sorted({0.0, *map(float, repair_probabilities)})
    eligible = [
        threshold
        for threshold in thresholds
        if evaluate_selector(
            labels,
            repair_probabilities >= threshold,
        )["repair_recall"]
        >= minimum_repair_recall
    ]
    return max(eligible)


# Evaluate transparent heuristic policies beside the learned selector.
def baseline_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    feature_schema: SelectorFeatureSchema = BASELINE_FEATURE_SCHEMA,
) -> dict[str, dict[str, float | int]]:
    feature_names = feature_schema.feature_names
    distance = features[:, feature_names.index("nearest_changed_block_distance")]
    overlap_columns = [
        feature_names.index(name)
        for name in (
            "changed_candidate_token_overlap_ratio",
            "introduced_candidate_token_overlap_ratio",
            "removed_candidate_token_overlap_ratio",
            "changed_candidate_token_jaccard",
        )
    ]
    return {
        "always_repair": evaluate_selector(labels, np.ones(len(labels), dtype=bool)),
        "radius_1": evaluate_selector(labels, distance <= 1),
        "keyword_overlap": evaluate_selector(
            labels,
            np.max(features[:, overlap_columns], axis=1) > 0,
        ),
    }


# Report the reuse available at several increasingly strict safety targets.
def operating_points(
    labels: np.ndarray,
    repair_probabilities: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    points = {}
    for recall_target in SAFETY_RECALL_TARGETS:
        threshold = select_repair_threshold(
            labels,
            repair_probabilities,
            minimum_repair_recall=recall_target,
        )
        points[f"{recall_target:.2f}"] = {
            "threshold": threshold,
            **evaluate_selector(labels, repair_probabilities >= threshold),
        }
    return points


# Rank blocks by repair score and report safety at fixed reuse budgets.
def reuse_budget_points(
    labels: np.ndarray,
    repair_probabilities: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    if (
        labels.ndim != 1
        or repair_probabilities.ndim != 1
        or len(labels) != len(repair_probabilities)
        or not len(labels)
        or not np.all(np.isfinite(repair_probabilities))
    ):
        raise ValueError("reuse-budget inputs must be aligned finite vectors")
    ordered = np.argsort(repair_probabilities, kind="stable")
    points = {}
    for target in REUSE_BUDGETS:
        count = math.ceil(target * len(labels))
        boundary = repair_probabilities[ordered[count - 1]]
        threshold = float(np.nextafter(boundary, np.inf))
        metrics = evaluate_selector(labels, repair_probabilities >= threshold)
        reused = metrics["safe_reuse"] + metrics["missed_repair"]
        points[f"{target:.2f}"] = {
            "target_reuse_rate": target,
            "threshold": threshold,
            "actual_reuse_rate": reused / len(labels),
            "unsafe_reuse_rate": metrics["missed_repair"] / reused,
            **metrics,
        }
    return points


# Train on one split and select the operating threshold only on validation.
def train_logistic_selector(
    dataset_path: Path,
    *,
    minimum_repair_recall: float = 0.95,
    evaluate_test: bool = False,
    feature_schema: SelectorFeatureSchema = BASELINE_FEATURE_SCHEMA,
) -> tuple[Pipeline, dict[str, object]]:
    dataset = load_selector_dataset(dataset_path, feature_schema=feature_schema)
    train_features, train_labels = dataset["train"]
    validation_features, validation_labels = dataset["validation"]
    model = logistic_model()
    model.fit(train_features, train_labels)
    validation_probabilities = model.predict_proba(validation_features)[:, 1]
    threshold = select_repair_threshold(
        validation_labels,
        validation_probabilities,
        minimum_repair_recall=minimum_repair_recall,
    )

    classifier = model.named_steps["classifier"]
    report: dict[str, object] = {
        "dataset": str(dataset_path),
        "feature_schema": feature_schema.name,
        "feature_names": list(feature_schema.feature_names),
        "minimum_validation_repair_recall": minimum_repair_recall,
        "selected_threshold": threshold,
        "coefficients": {
            name: float(weight)
            for name, weight in zip(
                feature_schema.feature_names,
                classifier.coef_[0],
                strict=True,
            )
        },
        "validation": {
            "logistic_regression": evaluate_selector(
                validation_labels,
                validation_probabilities >= threshold,
                repair_probabilities=validation_probabilities,
            ),
            "baselines": baseline_metrics(
                validation_features,
                validation_labels,
                feature_schema=feature_schema,
            ),
            "operating_points": operating_points(
                validation_labels,
                validation_probabilities,
            ),
            "reuse_budget_points": reuse_budget_points(
                validation_labels,
                validation_probabilities,
            ),
        },
    }
    if evaluate_test:
        test_features, test_labels = dataset["test"]
        test_probabilities = model.predict_proba(test_features)[:, 1]
        report["test"] = {
            "logistic_regression": evaluate_selector(
                test_labels,
                test_probabilities >= threshold,
                repair_probabilities=test_probabilities,
            ),
            "baselines": baseline_metrics(
                test_features,
                test_labels,
                feature_schema=feature_schema,
            ),
        }
    return model, report


# Export logistic regression through the existing one-layer runtime format.
def export_logistic_selector(
    model: Pipeline,
    report: dict[str, object],
    output_path: Path,
) -> dict[str, object]:
    scaler = model.named_steps["scale"]
    classifier = model.named_steps["classifier"]
    artifact = {
        "schema_version": 1,
        "model_type": "standard_scaler_mlp_binary_repair_selector",
        "feature_schema": report["feature_schema"],
        "feature_names": report["feature_names"],
        "selected_repair_threshold": report["selected_threshold"],
        "reuse_budget_thresholds": {
            target: point["threshold"]
            for target, point in report["validation"][
                "reuse_budget_points"
            ].items()
        },
        "standardizer": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "layers": [
            {
                "weights": classifier.coef_.T.tolist(),
                "bias": classifier.intercept_.tolist(),
                "activation": "logistic",
            }
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


# Train the baseline and write its JSON report without persisting the model yet.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path)
    parser.add_argument("--minimum-repair-recall", type=float, default=0.95)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--feature-schema",
        choices=sorted(FEATURE_SCHEMAS),
        default=BASELINE_FEATURE_SCHEMA.name,
    )
    args = parser.parse_args()

    model, report = train_logistic_selector(
        args.dataset,
        minimum_repair_recall=args.minimum_repair_recall,
        evaluate_test=args.evaluate_test,
        feature_schema=selector_feature_schema(args.feature_schema),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.runtime_output:
        export_logistic_selector(model, report, args.runtime_output)
    validation = report["validation"]["logistic_regression"]
    print(f"Selected repair threshold: {report['selected_threshold']:.6f}")
    print(f"Validation repair recall: {validation['repair_recall']:.3f}")
    print(f"Validation selected reuse rate: {validation['selected_reuse_rate']:.3f}")
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
