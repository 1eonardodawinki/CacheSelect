"""Explain how cheap selector features behave on natural causal labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cacheselect.selector_features import (
    CONTEXT_FEATURE_SCHEMA,
    FEATURE_SCHEMAS,
    SelectorFeatureSchema,
    selector_feature_schema,
)


# Parse the labelled natural rows while retaining their transition identity.
def _load_rows(
    path: Path,
    feature_schema: SelectorFeatureSchema,
) -> list[tuple[str, bool, np.ndarray]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as input_file:
        for line_number, row in enumerate(csv.DictReader(input_file), start=2):
            transition_id = row.get("transition_id", "").strip()
            decision = row.get("decision")
            if not transition_id or decision not in {"repair", "reuse"}:
                raise ValueError(f"line {line_number}: invalid transition or decision")
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
            rows.append((transition_id, decision == "repair", np.asarray(values)))
    if not rows:
        raise ValueError("natural selector dataset is empty")
    return rows


# Compare each feature globally and within transitions that contain both labels.
def analyze_natural_selector_failures(
    path: Path,
    *,
    feature_schema: SelectorFeatureSchema = CONTEXT_FEATURE_SCHEMA,
) -> dict[str, object]:
    rows = _load_rows(path, feature_schema)
    features = np.vstack([row[2] for row in rows])
    labels = np.asarray([row[1] for row in rows], dtype=bool)
    if not np.any(labels) or not np.any(~labels):
        raise ValueError("failure analysis requires both repair and reuse labels")
    grouped: dict[str, list[int]] = defaultdict(list)
    for row_index, (transition_id, _, _) in enumerate(rows):
        grouped[transition_id].append(row_index)

    transition_summaries = []
    for transition_id, indices in sorted(grouped.items()):
        transition_labels = labels[indices]
        transition_summaries.append(
            {
                "transition_id": transition_id,
                "examples": len(indices),
                "repair": int(np.sum(transition_labels)),
                "reuse": int(np.sum(~transition_labels)),
            }
        )
    mixed_groups = [
        indices
        for indices in grouped.values()
        if np.any(labels[indices]) and np.any(~labels[indices])
    ]

    feature_summaries = []
    for column, feature_name in enumerate(feature_schema.feature_names):
        repair_values = features[labels, column]
        reuse_values = features[~labels, column]
        scale = float(np.std(features[:, column]))
        directions = []
        for indices in mixed_groups:
            group_labels = labels[indices]
            group_values = features[indices, column]
            difference = float(
                np.mean(group_values[group_labels])
                - np.mean(group_values[~group_labels])
            )
            directions.append(int(np.sign(difference)))
        feature_summaries.append(
            {
                "feature": feature_name,
                "repair_median": float(np.median(repair_values)),
                "reuse_median": float(np.median(reuse_values)),
                "standardized_mean_gap": (
                    float((np.mean(repair_values) - np.mean(reuse_values)) / scale)
                    if scale > 0
                    else 0.0
                ),
                "mixed_transition_repair_higher": directions.count(1),
                "mixed_transition_repair_lower": directions.count(-1),
                "mixed_transition_ties": directions.count(0),
            }
        )
    feature_summaries.sort(
        key=lambda item: abs(float(item["standardized_mean_gap"])), reverse=True
    )
    repairs_by_transition = [summary["repair"] for summary in transition_summaries]
    return {
        "schema_version": 1,
        "analysis": "natural-selector-feature-failures",
        "feature_schema": feature_schema.name,
        "input": str(path),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "examples": len(rows),
        "transitions": len(grouped),
        "repair_labels": int(np.sum(labels)),
        "reuse_labels": int(np.sum(~labels)),
        "mixed_label_transitions": len(mixed_groups),
        "largest_transition_share_of_repairs": max(repairs_by_transition)
        / int(np.sum(labels)),
        "transition_summaries": transition_summaries,
        "feature_summaries": feature_summaries,
    }


# Run the diagnostic and save a provenance-bound JSON artifact.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-schema",
        choices=sorted(FEATURE_SCHEMAS),
        default=CONTEXT_FEATURE_SCHEMA.name,
    )
    args = parser.parse_args()
    report = analyze_natural_selector_failures(
        args.input,
        feature_schema=selector_feature_schema(args.feature_schema),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Analyzed {report['examples']} blocks across "
        f"{report['transitions']} transitions"
    )


if __name__ == "__main__":
    main()
