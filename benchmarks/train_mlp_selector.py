"""Train a deterministic small neural candidate-block repair selector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
    select_repair_threshold,
)


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


# Train a small deterministic MLP and select its safety threshold on validation.
def train_mlp_selector(
    dataset_path: Path,
    *,
    minimum_repair_recall: float = 0.95,
    feature_schema: SelectorFeatureSchema = BASELINE_FEATURE_SCHEMA,
) -> tuple[Pipeline, dict[str, object]]:
    dataset = load_selector_dataset(dataset_path, feature_schema=feature_schema)
    train_features, train_labels = dataset["train"]
    validation_features, validation_labels = dataset["validation"]
    balanced_features, balanced_labels = balanced_binary_training_rows(
        train_features,
        train_labels,
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(8,),
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
            "hidden_layer_sizes": [8],
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
        },
    }
    return model, report
