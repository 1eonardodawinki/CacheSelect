# MTRAG natural block-coverage gate

This artifact records the CPU-only gate run before spending GPU time on causal
labels. It uses the human MTRAG full-RAG tasks exactly as released and the
versioned natural CacheSelect prompt renderer. No passage padding or artificial
block alignment was applied.

## Reproduction

Source:

- `IBM/mt-rag-benchmark/mtrag-human/generation_tasks/RAG.jsonl`
- Git revision `cc5b1d481b391181b89f7ced860308482e785463`

Command:

```bash
python -m benchmarks.analyze_mtrag_coverage \
  --input /path/to/RAG.jsonl \
  --output /tmp/mtrag-coverage.json \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --block-size 16 \
  --local-files-only
```

The complete per-transition JSON is approximately 11 MB and is reproducible
from the public input, so only its compact audited summary is committed.

## Gate result

| Measure | Result |
| --- | ---: |
| Tasks | 842 |
| Conversations | 110 |
| Adjacent transitions | 732 |
| Transitions sharing a retrieved passage | 285 |
| Transitions with any exact candidate block | 601 |
| Transitions with a physically aligned source block | 159 |
| Aligned-source transitions that also share a passage | 106 |
| All exact candidate blocks | 16,642 |
| Physically aligned source blocks | 1,215 |
| Candidates requiring gathering or repacking | 15,427 |

Among the 106 aligned-source transitions with a shared document, the current
prompt length ranges from 881 to 3,960 tokens. The median transition contains
three aligned source blocks and the maximum contains 64. These cases fit the
current 4,096-token experiment envelope.

## Interpretation

The gate passes: natural MTRAG requests provide enough executable cases for a
counterfactual pilot without manufacturing block alignment. The result also
exposes the present runtime limit. Only 1,215 of 16,642 content-identical
candidate blocks begin at a compatible source block boundary; the other
15,427 need future gathering, repacking, or finer-grained reuse.

The 601 content-match transitions include repeated history and prompt
boilerplate, not only retrieved documents. The first GPU pilot therefore uses
the stricter 106-transition pool that has both an aligned source candidate and
an explicitly shared MTRAG document ID. Block safety is still unknown until
the forced single-block counterfactual runner creates a causal label.
