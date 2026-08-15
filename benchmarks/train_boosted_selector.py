"""Train a compact gradient-boosted candidate-block repair selector."""

from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

from benchmarks.train_logistic_selector import (
    FEATURE_NAMES,
    evaluate_selector,
    load_selector_dataset,
    operating_points,
    select_repair_threshold,
)


# Train one compact tree ensemble and choose its validation operating point.
def train_boosted_selector(
    dataset_path: Path,
    *,
    minimum_repair_recall: float = 0.95,
) -> tuple[HistGradientBoostingClassifier, dict[str, object]]:
    dataset = load_selector_dataset(dataset_path)
    train_features, train_labels = dataset["train"]
    validation_features, validation_labels = dataset["validation"]
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        class_weight="balanced",
        random_state=0,
    )
    model.fit(train_features, train_labels)
    validation_probabilities = model.predict_proba(validation_features)[:, 1]
    threshold = select_repair_threshold(
        validation_labels,
        validation_probabilities,
        minimum_repair_recall=minimum_repair_recall,
    )
    report: dict[str, object] = {
        "dataset": str(dataset_path),
        "model": "hist_gradient_boosting",
        "feature_names": list(FEATURE_NAMES),
        "minimum_validation_repair_recall": minimum_repair_recall,
        "selected_threshold": threshold,
        "hyperparameters": {
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "class_weight": "balanced",
        },
        "validation": {
            "hist_gradient_boosting": evaluate_selector(
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
