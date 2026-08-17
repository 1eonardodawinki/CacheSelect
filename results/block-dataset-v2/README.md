# Candidate-block context dataset v2

This bundle regenerates the controlled `block-dataset-v1` matrix with the same
labels, splits, tokenizer, and original 17 selector features. It appends ten
prompt-context features defined by the `block-context-v2` schema.

## Contents

- `candidate-blocks.csv`: 4,520 labelled candidate blocks with 27 model inputs.
- `baseline-v1-selector-validation-comparison.json`: current three-model
  validation results using only the original 17 features.
- `selector-validation-comparison.json`: the same comparison using all 27
  features.

Generator commit: `cfb3a22d0d980d34e1b526baf8e99c8985f5a0a1`.

SHA-256:

- Dataset: `cd6a27beda68b939bd70ba228238d523d10372d2080a31b32576046ed2468348`
- v1 comparison: `b3a0122f36f9a2fca6a8a4afd6ee47741a88dca240f31122f59800ca4e0b032d`
- v2 comparison: `f608b9006a50189a31fa0b1092eafec6bcb82f201394a50681b701bca4c7ac40`

## Reproduction

```bash
python -m benchmarks.generate_block_dataset \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --feature-schema block-context-v2 \
  --output results/block-dataset-v2/candidate-blocks.csv

python -m benchmarks.compare_selectors \
  --dataset results/block-dataset-v2/candidate-blocks.csv \
  --feature-schema block-context-v2 \
  --minimum-repair-recall 0.95 \
  --output results/block-dataset-v2/selector-validation-comparison.json
```

## Dataset counts

| Split | REPAIR | REUSE | Total |
|---|---:|---:|---:|
| Train | 577 | 2,443 | 3,020 |
| Validation | 140 | 605 | 745 |
| Test | 145 | 610 | 755 |

The test split remains unevaluated. Validation uses the held-out `profile`
wording family, while the `channel` family remains sealed for final evaluation.

## Synthetic validation comparison

| Model | v1 reuse at >=95% repair recall | v2 reuse at >=95% repair recall | v2 reuse at 100% repair recall |
|---|---:|---:|---:|
| Logistic regression | 42.1% | 81.2% | 78.5% |
| Histogram gradient boosting | 62.8% | 81.1% | 80.1% |
| MLP | 67.2% | 80.0% | 78.3% |

These results show that the context features separate the controlled synthetic
dependencies much more effectively. They do not establish generalization to
natural prompts; the separately collected MTRAG counterfactual labels are the
next out-of-domain check.
