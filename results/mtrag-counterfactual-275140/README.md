# Expanded MTRAG counterfactual labels

This bundle records Slurm job `275140`, which executed isolated block-reuse
interventions for the frozen `mtrag-expanded-v1` manifest.

## Execution

- Project commit: `d5397dd41788f5c2a6d1d5e1f56a679a91287f10`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- vLLM: `0.0.0+d6dbdb9b0`
- GPU: NVIDIA A16 (15,356 MiB)
- Cases: 8
- Block interventions: 53
- Invalid executions: 0

Every trial used a fresh uncached reference and donor, then reused one selected
block while repairing all other candidate blocks. The full request ledger
contains 175 completed request lifecycles and no failures.

## Adjudication

- 26 interventions exactly matched their reference output.
- 27 non-identical pairs entered blinded semantic review.
- Blinded review produced 16 REUSE, 10 REPAIR, and 1 unresolved decision.
- Final model table: 52 rows (42 REUSE and 10 REPAIR).

The unresolved agriculture comparison remains in the review audit but is
excluded from the training table. It was not forced into either class.

`mtrag-curated-blocks.csv` uses the 27-input `block-context-v2` schema. Its
SHA-256 is
`8f0250ebe186add4224ce68d68996eeb836836936ff839eeb7978b0de7bfacf6`.

## Raw evidence

The append-only request ledger is retained locally and on Imperial storage but
is not committed because it is approximately 10 MB. Its SHA-256 is
`0ba7834e8b7903bf65d0fab1e690480248f5272bcb58a97b372d601b66b407f8`.
The compact committed artifacts retain this hash so the ledger can be verified.
