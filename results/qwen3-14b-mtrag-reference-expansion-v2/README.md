# Qwen3-14B MTRAG reference expansion v2

This frozen selection contains every remaining train or validation transition
that has at least one executable 16-token reuse candidate and a prompt of at
most 7,000 Qwen3-14B tokens.

- 21 training transitions
- 15 validation transitions
- 36 transitions total
- test conversations remain sealed

Together with the 102 previously evaluated reference tasks, these manifests
cover all 138 eligible non-test transitions. The remaining rows are
intentionally domain-skewed because the earlier balanced selections had
already consumed every other eligible row.

Generate their full-computation answers before adding any of them to the
counterfactual block plan. Only references that pass the same blinded review
used for the earlier batches should become causal experiments.
