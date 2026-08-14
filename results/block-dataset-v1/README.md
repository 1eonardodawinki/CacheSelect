# Candidate-block dependency dataset v1

This bundle contains the first model-tokenized supervised dataset for the
CacheSelect block selector. It is synthetic by design: each trace changes one
selector while leaving its two mapping facts and final query text unchanged.
The benchmark generator therefore knows which unchanged blocks require repair.

## Contents

- `candidate-blocks.csv`: 4,520 candidate KV-block rows tokenized with
  `Qwen/Qwen2.5-1.5B-Instruct` at block size 16.
- `logistic-validation.json`: first class-balanced logistic-regression report;
  the held-out test split was deliberately not evaluated.
- Generator commit: `1fbfb46eb385f415cc04792974ea931cf0c5a007`.
- SHA-256: `406f725da9168acc3249ac827db1578098c1363a23af8aa065d3258633e8bed1`.

The dataset was generated with:

```bash
python -m benchmarks.generate_block_dataset \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output results/block-dataset-v1/candidate-blocks.csv
```

## Split and label counts

| Split | REPAIR | REUSE | Total |
|---|---:|---:|---:|
| Train | 577 | 2,443 | 3,020 |
| Validation | 140 | 605 | 745 |
| Test | 145 | 610 | 755 |

The split is by complete wording family, not by individual block. The `profile`
family is reserved for validation and the `channel` family for testing. This
prevents near-identical blocks from the same trace family appearing in multiple
partitions.

## Position-bias control

Dependent facts appear in five layouts: early, middle, late, separated by
neutral text, and reversed. Across training rows, mean normalized prompt
position is 0.734 for `REPAIR` and 0.640 for `REUSE`. A position-only ranking
has ROC AUC 0.613, reduced from 0.847 in the discarded fixed-layout preview.

## First classifier baseline

At the selected 95% minimum validation repair-recall target, logistic
regression reaches 0.957 repair recall and selects 0.421 of candidate blocks
for reuse. Of the selected reuse blocks, 0.981 are correctly labelled safe.
Its validation ROC AUC is 0.917 and average precision is 0.839.

The stricter operating-point analysis is important: at the current 99% and
100% repair-recall targets, the model selects no reuse. This is an explicit
limitation of the current features rather than a hidden positive result. The
test family remains untouched until model and feature selection are frozen.

## Limitation

These are deterministic synthetic dependency labels, not measured causal
labels from arbitrary real prompts. This dataset is appropriate for initial
classifier development and feature ablations. Final claims must also use
counterfactual execution and held-out workloads.
