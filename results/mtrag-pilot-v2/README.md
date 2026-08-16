# Audited MTRAG counterfactual pilot

This manifest admits only tasks marked `pass` in
`results/mtrag-quality-calibration-v1/manual-audit-v1.json`. The earlier
geometry-only selection was discarded after reference calibration found that
two of its four current answers were semantically wrong.

The pilot contains one natural adjacent RAG transition from each of the four
MTRAG collections. Their live plans contain 88 testable aligned KV blocks, but
only 7 targets were selected before intervention outcomes were observed. For
each target, every other candidate is still repaired; target sampling reduces
GPU work without weakening the one-variable counterfactual.

## Reproduction

The pure `select_audited_mtrag_counterfactual_pilot` function applies the
frozen audit to the coverage artifact. It deterministically ranks transitions
and target blocks before any counterfactual outputs are observed.

The manifest records hashes for the coverage, audit, reference output, and raw
MTRAG source. A GPU runner must validate the complete candidate and testable
lists against live vLLM discovery before running only `target_block_indices`.
