"""Train a deterministic small neural candidate-block repair selector."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cacheselect.selector_features import (
    BASELINE_FEATURE_SCHEMA,
    SelectorFeatureSchema,
)
from benchmarks.train_logistic_selector import (
    evaluate_selector,
    load_selector_dataset,
    operating_points,
    reuse_budget_points,
    select_repair_threshold,
)

MLP_ARCHITECTURES = ((4,), (8,), (16,), (32,), (16, 8))


# Repeat the minority class deterministically for scikit-learn 1.4 compatibility.
def balanced_binary_training_rows(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("training features and labels have incompatible shapes")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("balanced training requires both binary classes")
    random = np.random.default_rng(random_state)
    class_indices = {value: np.flatnonzero(labels == value) for value in (0, 1)}
    minority_value = min(class_indices, key=lambda value: len(class_indices[value]))
    minority_indices = class_indices[minority_value]
    majority_count = max(len(indices) for indices in class_indices.values())
    extra = random.choice(
        minority_indices,
        size=majority_count - len(minority_indices),
        replace=True,
    )
    # Freeze class-grouped ordering because Adam's mini-batches depend on it.
    order = np.concatenate((class_indices[0], class_indices[1], extra))
    random.shuffle(order)
    return features[order], labels[order]


def _mlp_pipeline(hidden_layer_sizes: tuple[int, ...]) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    learning_rate_init=0.003,
                    max_iter=2000,
                    tol=1e-4,
                    n_iter_no_change=40,
                    early_stopping=False,
                    random_state=0,
                ),
            ),
        ]
    )


# Train a small deterministic MLP and select its safety threshold on validation.
def train_mlp_selector(
    dataset_path: Path,
    *,
    minimum_repair_recall: float = 0.95,
    feature_schema: SelectorFeatureSchema = BASELINE_FEATURE_SCHEMA,
    hidden_layer_sizes: tuple[int, ...] = (8,),
) -> tuple[Pipeline, dict[str, object]]:
    dataset = load_selector_dataset(dataset_path, feature_schema=feature_schema)
    train_features, train_labels = dataset["train"]
    validation_features, validation_labels = dataset["validation"]
    balanced_features, balanced_labels = balanced_binary_training_rows(
        train_features,
        train_labels,
    )
    model = _mlp_pipeline(hidden_layer_sizes)
    model.fit(balanced_features, balanced_labels)
    validation_probabilities = model.predict_proba(validation_features)[:, 1]
    threshold = select_repair_threshold(
        validation_labels,
        validation_probabilities,
        minimum_repair_recall=minimum_repair_recall,
    )
    classifier = model.named_steps["classifier"]
    report: dict[str, object] = {
        "dataset": str(dataset_path),
        "model": "mlp",
        "feature_schema": feature_schema.name,
        "feature_names": list(feature_schema.feature_names),
        "minimum_validation_repair_recall": minimum_repair_recall,
        "selected_threshold": threshold,
        "test_split_evaluated": False,
        "hyperparameters": {
            "hidden_layer_sizes": list(hidden_layer_sizes),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.001,
            "learning_rate_init": 0.003,
            "max_iter": 2000,
            "tol": 1e-4,
            "n_iter_no_change": 40,
            "early_stopping": False,
            "random_state": 0,
            "class_balancing": "deterministic_minority_oversampling",
        },
        "training": {
            "original_examples": len(train_labels),
            "balanced_examples": len(balanced_labels),
            "balanced_repair_examples": int(np.sum(balanced_labels == 1)),
            "balanced_reuse_examples": int(np.sum(balanced_labels == 0)),
            "iterations": classifier.n_iter_,
            "converged": classifier.n_iter_ < classifier.max_iter,
        },
        "validation": {
            "mlp": evaluate_selector(
                validation_labels,
                validation_probabilities >= threshold,
                repair_probabilities=validation_probabilities,
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
    return model, report


# Save only the values needed for dependency-free runtime inference.
def export_mlp_selector(
    model: Pipeline,
    report: dict[str, object],
    output_path: Path,
) -> dict[str, object]:
    scaler = model.named_steps["scale"]
    classifier = model.named_steps["classifier"]
    activations = [classifier.activation] * (len(classifier.coefs_) - 1)
    activations.append(classifier.out_activation_)
    artifact = {
        "schema_version": 1,
        "model_type": "standard_scaler_mlp_binary_repair_selector",
        "feature_schema": report["feature_schema"],
        "feature_names": report["feature_names"],
        "selected_repair_threshold": report["selected_threshold"],
        "operating_point_thresholds": {
            target: values["threshold"]
            for target, values in report["validation"]["operating_points"].items()
        },
        "reuse_budget_thresholds": {
            target: values["threshold"]
            for target, values in report["validation"][
                "reuse_budget_points"
            ].items()
        },
        "standardizer": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "layers": [
            {
                "weights": weights.tolist(),
                "bias": bias.tolist(),
                "activation": activation,
            }
            for weights, bias, activation in zip(
                classifier.coefs_, classifier.intercepts_, activations, strict=True
            )
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def _training_groups(path: Path) -> np.ndarray:
    with path.open(newline="") as input_file:
        groups = [
            row.get("mtrag_conversation_id", "")
            for row in csv.DictReader(input_file)
            if row.get("split") == "train"
        ]
    if not groups or not all(groups):
        raise ValueError("training rows require mtrag_conversation_id")
    return np.asarray(groups)


# Compare small MLPs using out-of-fold predictions from unseen conversations.
def sweep_mlp_architectures(
    dataset_path: Path,
    *,
    minimum_repair_recall: float = 0.95,
    feature_schema: SelectorFeatureSchema = BASELINE_FEATURE_SCHEMA,
    architectures: tuple[tuple[int, ...], ...] = MLP_ARCHITECTURES,
    folds: int = 5,
) -> dict[str, object]:
    features, labels = load_selector_dataset(
        dataset_path, feature_schema=feature_schema
    )["train"]
    groups = _training_groups(dataset_path)
    if len(groups) != len(labels):
        raise ValueError("training groups and examples are misaligned")
    results = []
    for architecture in architectures:
        probabilities = np.empty(len(labels))
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=0
        )
        for train, held_out in splitter.split(features, labels, groups):
            balanced = balanced_binary_training_rows(features[train], labels[train])
            model = _mlp_pipeline(architecture)
            model.fit(*balanced)
            probabilities[held_out] = model.predict_proba(features[held_out])[:, 1]
        threshold = select_repair_threshold(
            labels,
            probabilities,
            minimum_repair_recall=minimum_repair_recall,
        )
        results.append(
            {
                "hidden_layer_sizes": list(architecture),
                "selected_threshold": threshold,
                "metrics": evaluate_selector(
                    labels,
                    probabilities >= threshold,
                    repair_probabilities=probabilities,
                ),
            }
        )
    recommended = max(results, key=lambda result: result["metrics"]["safe_reuse"])
    return {
        "dataset": str(dataset_path),
        "evaluation": f"{folds}-fold conversation-grouped cross-validation",
        "minimum_repair_recall": minimum_repair_recall,
        "feature_schema": feature_schema.name,
        "training_examples": len(labels),
        "conversation_groups": len(set(groups)),
        "architectures": results,
        "recommended_hidden_layer_sizes": recommended["hidden_layer_sizes"],
    }
