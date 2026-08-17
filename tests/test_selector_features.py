from unittest import TestCase

from cacheselect.selector_features import (
    BASELINE_FEATURE_SCHEMA,
    CONTEXT_FEATURE_SCHEMA,
    CONTEXT_NUMERIC_FEATURES,
    selector_feature_schema,
)


class SelectorFeatureSchemaTests(TestCase):
    # Preserve the published baseline while extending it with ten context signals.
    def test_versions_feature_contracts(self):
        self.assertEqual(len(BASELINE_FEATURE_SCHEMA.feature_names), 17)
        self.assertEqual(len(CONTEXT_NUMERIC_FEATURES), 10)
        self.assertEqual(len(CONTEXT_FEATURE_SCHEMA.feature_names), 27)
        self.assertEqual(
            CONTEXT_FEATURE_SCHEMA.feature_names[:17],
            BASELINE_FEATURE_SCHEMA.feature_names,
        )
        self.assertEqual(
            selector_feature_schema("block-context-v2"),
            CONTEXT_FEATURE_SCHEMA,
        )

    # Refuse misspelled or future schema names instead of changing model inputs.
    def test_rejects_unknown_schema(self):
        with self.assertRaisesRegex(ValueError, "unknown selector feature schema"):
            selector_feature_schema("block-v3")
