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
- `mtrag-transfer-evaluation-v1.json`: frozen 17-feature models evaluated on
  the natural MTRAG causal labels.
- `mtrag-transfer-evaluation.json`: frozen 27-feature models evaluated on the
  same natural labels.

Generator commit: `cfb3a22d0d980d34e1b526baf8e99c8985f5a0a1`.

SHA-256:

- Dataset: `cd6a27beda68b939bd70ba228238d523d10372d2080a31b32576046ed2468348`
- v1 comparison: `b3a0122f36f9a2fca6a8a4afd6ee47741a88dca240f31122f59800ca4e0b032d`
- v2 comparison: `f608b9006a50189a31fa0b1092eafec6bcb82f201394a50681b701bca4c7ac40`
- MTRAG v1 transfer: `a363063b6cea8a3be632394f157263b442e8469c84103e094eb398dad6affee8`
- MTRAG v2 transfer: `c8514cfacccd4567539f53d20cb78dc5cad50917de9c995b3731ceac02cf2268`
- Natural feature analysis: `eafcdad2f0af1458ab36b6221b2118ef24d309347896aeae5fabe9ec40b37efa`

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

python -m benchmarks.evaluate_selector_transfer \
  --source-dataset results/block-dataset-v2/candidate-blocks.csv \
  --target-dataset results/mtrag-counterfactual-275140/mtrag-curated-blocks.csv \
  --feature-schema block-context-v2 \
  --minimum-repair-recall 0.95 \
  --output results/block-dataset-v2/mtrag-transfer-evaluation.json

python -m benchmarks.analyze_natural_selector_failures \
  --input results/mtrag-counterfactual-275140/mtrag-curated-blocks.csv \
  --feature-schema block-context-v2 \
  --output results/block-dataset-v2/mtrag-natural-feature-analysis.json
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

## Natural MTRAG transfer

The three models were trained only on synthetic rows, kept at their synthetic
validation thresholds, and then evaluated on 52 natural counterfactual labels
(42 REUSE and 10 REPAIR). MTRAG was not used for fitting or threshold choice.

| Schema and model | Repairs found | Repairs missed | Safe blocks reused |
|---|---:|---:|---:|
| v1 logistic | 1/10 | 9 | 42/42 |
| v1 boosted tree | 10/10 | 0 | 2/42 |
| v1 MLP | 5/10 | 5 | 2/42 |
| v2 logistic | 1/10 | 9 | 42/42 |
| v2 boosted tree | 6/10 | 4 | 5/42 |
| v2 MLP | 0/10 | 10 | 42/42 |

The synthetic improvement does not transfer safely. The v1 boosted tree is the
only zero-miss learned policy on this small natural sample, but it is nearly as
conservative as always repairing. These results rule out deploying the current
synthetic-trained models and motivate calibration with natural causal labels.

## Natural failure analysis

`mtrag-natural-feature-analysis.json` compares all 27 features on the 52 causal
MTRAG labels without fitting another model. The rows come from only seven
transitions, and eight of the ten REPAIR labels belong to one transition. Only
two transitions contain both REPAIR and REUSE blocks.

Some prompt-level features therefore show large global differences while being
constant inside a transition. Total changed-token counts can identify the prompt
that produced most failures, but cannot identify its individual unsafe blocks.
Edit spans before the candidate and changed-token Jaccard show some
within-transition separation, but two mixed transitions are far too little
evidence for a deployable rule. Future data must be split and evaluated by whole
transition, never by randomly mixing blocks from the same prompt.
