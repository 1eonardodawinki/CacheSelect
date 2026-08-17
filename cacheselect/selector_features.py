"""Versioned feature contracts shared by every CacheSelect model."""

from __future__ import annotations

from dataclasses import dataclass, fields

from cacheselect.context_features import CandidateContextFeatures


@dataclass(frozen=True)
class SelectorFeatureSchema:
    """One ordered model-input contract for training and serving."""

    name: str
    feature_names: tuple[str, ...]
    boolean_features: tuple[str, ...]

    # Return non-boolean columns without changing their relative order.
    @property
    def numeric_features(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.feature_names if name not in self.boolean_features
        )


BASELINE_NUMERIC_FEATURES = (
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
BASELINE_BOOLEAN_FEATURES = ("same_position_match", "requires_repacking")
CONTEXT_NUMERIC_FEATURES = tuple(
    field.name for field in fields(CandidateContextFeatures)
)

BASELINE_FEATURE_SCHEMA = SelectorFeatureSchema(
    "block-v1",
    BASELINE_NUMERIC_FEATURES + BASELINE_BOOLEAN_FEATURES,
    BASELINE_BOOLEAN_FEATURES,
)
CONTEXT_FEATURE_SCHEMA = SelectorFeatureSchema(
    "block-context-v2",
    BASELINE_FEATURE_SCHEMA.feature_names + CONTEXT_NUMERIC_FEATURES,
    BASELINE_BOOLEAN_FEATURES,
)
FEATURE_SCHEMAS = {
    schema.name: schema for schema in (BASELINE_FEATURE_SCHEMA, CONTEXT_FEATURE_SCHEMA)
}

# Backward-compatible aliases retain the current 17-feature default.
NUMERIC_FEATURES = BASELINE_FEATURE_SCHEMA.numeric_features
BOOLEAN_FEATURES = BASELINE_FEATURE_SCHEMA.boolean_features
FEATURE_NAMES = BASELINE_FEATURE_SCHEMA.feature_names


# Resolve a command-line schema name without silently selecting a fallback.
def selector_feature_schema(name: str) -> SelectorFeatureSchema:
    try:
        return FEATURE_SCHEMAS[name]
    except KeyError as error:
        raise ValueError(f"unknown selector feature schema: {name!r}") from error
