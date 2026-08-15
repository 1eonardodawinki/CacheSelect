"""Train and evaluate a logistic candidate-block repair selector."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = (
    "previous_token_count",
    "current_token_count",
    "previous_changed_token_count",
    "current_changed_token_count",
    "block_size",
    "candidate_block_index",
    "candidate_position_ratio",
    "relative_block_offset",
    "nearest_changed_block_distance",
    "source_displacement_blocks",
    "candidate_share_of_native_recompute",
    "changed_candidate_token_overlap_ratio",
    "introduced_candidate_token_overlap_ratio",
    "removed_candidate_token_overlap_ratio",
    "changed_candidate_token_jaccard",
)
BOOLEAN_FEATURES = ("same_position_match", "requires_repacking")
FEATURE_NAMES = NUMERIC_FEATURES + BOOLEAN_FEATURES
SPLITS = ("train", "validation", "test")
SAFETY_RECALL_TARGETS = (0.90, 0.95, 0.99, 1.0)


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
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[list[float], int]]] = {
        split: [] for split in SPLITS
    }
    with path.open(newline="") as input_file:
        for line_number, row in enumerate(csv.DictReader(input_file), start=2):
            split = row.get("split")
            if split not in grouped:
                raise ValueError(f"line {line_number}: invalid split {split!r}")
            decision = row.get("decision")
            if decision not in ("repair", "reuse"):
                raise ValueError(f"line {line_number}: invalid decision {decision!r}")
            try:
                features = [float(row[name]) for name in NUMERIC_FEATURES]
                features.extend(
                    _parse_boolean(row[name], field_name=name)
                    for name in BOOLEAN_FEATURES
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"line {line_number}: invalid selector feature"
                ) from error
            grouped[split].append((features, int(decision == "repair")))

    dataset = {}
    for split, examples in grouped.items():
        if not examples:
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
            2 * repair_precision * repair_recall
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
) -> dict[str, dict[str, float | int]]:
    distance = features[:, FEATURE_NAMES.index("nearest_changed_block_distance")]
    overlap_columns = [
        FEATURE_NAMES.index(name)
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


# Train on one split and select the operating threshold only on validation.
def train_logistic_selector(
    dataset_path: Path,
    *,
    minimum_repair_recall: float = 0.95,
    evaluate_test: bool = False,
) -> tuple[Pipeline, dict[str, object]]:
    dataset = load_selector_dataset(dataset_path)
    train_features, train_labels = dataset["train"]
    validation_features, validation_labels = dataset["validation"]
    model = Pipeline(
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
        "feature_names": list(FEATURE_NAMES),
        "minimum_validation_repair_recall": minimum_repair_recall,
        "selected_threshold": threshold,
        "coefficients": {
            name: float(weight)
            for name, weight in zip(
                FEATURE_NAMES,
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
            ),
            "operating_points": operating_points(
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
            "baselines": baseline_metrics(test_features, test_labels),
        }
    return model, report


# Train the baseline and write its JSON report without persisting the model yet.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-repair-recall", type=float, default=0.95)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    _, report = train_logistic_selector(
        args.dataset,
        minimum_repair_recall=args.minimum_repair_recall,
        evaluate_test=args.evaluate_test,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    validation = report["validation"]["logistic_regression"]
    print(f"Selected repair threshold: {report['selected_threshold']:.6f}")
    print(f"Validation repair recall: {validation['repair_recall']:.3f}")
    print(f"Validation selected reuse rate: {validation['selected_reuse_rate']:.3f}")
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
