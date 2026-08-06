# Online locator calibration run 270693

This directory archives the first prompt-length and edit-position calibration
of CacheSelect's online aligned-block locator inside vLLM. The nine conditions
ran on Imperial NVIDIA A16 GPUs with `Qwen/Qwen2.5-1.5B-Instruct`, project
commit `2023302`, vLLM build `0.0.0+d6dbdb9b0`, APC enabled and a 16-token
block size.

## Design

The benchmark generated prompts close to 256, 1,024 and 4,096 tokens. At each
length it changed one same-length marker near the early, middle or late part of
the prompt. Every condition used a fresh vLLM process and sent two requests:
the cold source prompt followed by its edited version.

CacheSelect remained in shadow mode. Native APC executed normally; the online
locator only reported target-aligned blocks whose token content was present in
the source request and whose source KV blocks were still resident. All such
blocks still require context repair before their KV can be reused safely.

## Result

| Prompt | Edit | APC hit | Native recompute | Online candidates | Resident | Candidate share |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 256 | early | 32 | 224 | 208 | 208 | 92.9% |
| 256 | middle | 96 | 160 | 144 | 144 | 90.0% |
| 256 | late | 160 | 96 | 80 | 80 | 83.3% |
| 1,024 | early | 32 | 992 | 976 | 976 | 98.4% |
| 1,024 | middle | 352 | 672 | 656 | 656 | 97.6% |
| 1,024 | late | 672 | 352 | 336 | 336 | 95.5% |
| 4,096 | early | 32 | 4,064 | 4,048 | 4,048 | 99.6% |
| 4,096 | middle | 1,376 | 2,720 | 2,704 | 2,704 | 99.4% |
| 4,096 | late | 2,720 | 1,376 | 1,360 | 1,360 | 98.8% |

Across all nine conditions, the locator found 10,512 resident candidate tokens:
98.6% of the 10,656 tokens native APC recomputed. The online and offline
aligned-block counts agreed in every condition. Candidate volume grew with
prompt length and was largest when the edit occurred early, as expected.

This result validates discovery and residency tracking, not accelerated reuse
or output equivalence after repair. The next milestone is to repair selected
candidate KV states and compare the repaired execution against full prefill.

## Contents

- `analysis/`: validated JSON, CSV and Markdown summaries;
- `generated-traces/`: deterministic source/edit traces for all conditions;
- `results/`: full result JSON and provenance manifests;
- `request-logs/`: append-only full input/output ledgers;
- `server-logs/`: one vLLM log per condition;
- `slurm-logs/`: the three array-task transcripts.

Recreate the analysis from the repository root with:

```bash
python -m benchmarks.analyze_locator_calibration \
  --input-root results/locator-calibration-270693
```
