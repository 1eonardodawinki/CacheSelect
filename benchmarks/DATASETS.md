# CacheSelect dataset strategy

This note records the dataset decision made on 16 August 2026. CacheSelect
needs request transitions, not isolated prompts: a previous request provides
KV state and a current request changes, moves, inserts, or removes content.
None of the public datasets below contains block-level `REUSE` or `REPAIR`
labels. CacheSelect creates those labels with its forced single-block
counterfactual runner.

## Selected dataset roles

| Role | Dataset | Purpose |
| --- | --- | --- |
| Controlled training and ablations | CacheSelect synthetic traces | Balance prompt length, edit position, dependency type, and difficult repair cases. |
| Main real workload | IBM MTRAG Human | Train and evaluate on natural consecutive multi-turn RAG requests with changing retrieved passages. |
| Optional scale-up | NVIDIA ChatRAG-Bench | Add conversational RAG examples only if MTRAG yields too few naturally reusable blocks. |
| Direct predecessor comparison | CacheBlend MuSiQue and 2WikiMultiHopQA inputs | Compare with CacheBlend-style modular RAG and short-answer quality metrics. |
| Non-RAG generalisation | Zeta next-edit prediction | Test whether a selector learned on RAG transfers to edited code prompts. |
| APC control | Full-turn ShareGPT or Codex SWE-bench Pro traces | Verify that the planner chooses ordinary prefix reuse for append-only histories. |
| Load stress only | RAGPulse | Replay realistic arrivals and document popularity without using it for semantic labels. |

LongBench is not currently a separate core dataset because its most relevant
tasks overlap with MuSiQue and 2WikiMultiHopQA. It remains an optional held-out
source if the final evaluation needs another long-context domain.

## Why MTRAG is the main real workload

The current human `RAG.jsonl` release contains 842 request tasks from 110
conversations, giving 732 consecutive transitions. A local inspection of the
five retrieved passage IDs per turn found:

| Adjacent-turn relationship | Transitions |
| --- | ---: |
| At least one shared passage | 285 (38.9%) |
| Partially overlapping passage sets | 252 |
| Same passage set in a different order | 21 |
| Identical passage order | 12 |
| No shared passage | 447 |

All occurrences of a repeated passage ID contained the same text. MTRAG
therefore provides natural cases such as `[A, B, C, D, E]` becoming
`[C, F, A, G, H]`: retained passages move outside the exact prefix while the
question and conversation history also evolve.

These passage-level counts are only a first filter. They do not prove that the
rendered prompts contain reusable 16-token vLLM blocks. Prompt formatting can
shift token boundaries, so natural token-block coverage must be measured
before scheduling counterfactual GPU experiments.

## Experimental split and leakage rules

- Split MTRAG by complete conversation ID, never by candidate block row.
- Keep every augmented version of a source question or document family in the
  same split.
- Fit selector thresholds on training and validation data only.
- Report held-out MTRAG results as the main generalisation result.
- Keep CacheBlend inputs and Zeta outside the main training split so they can
  measure cross-dataset and cross-domain transfer.
- Keep append-only controls separate from partial-reuse accuracy averages;
  their success condition is selecting APC without a regression.

## Planned training comparison

Train the same selector family at three data operating points:

1. Controlled synthetic transitions only.
2. Causally labelled MTRAG transitions only.
3. Synthetic and MTRAG transitions combined.

Compare logistic regression, gradient-boosted trees, and the small MLP on the
same held-out conversations and at the same repair-recall operating points.
This distinguishes the benefit of controlled coverage from the benefit of
real request diversity.

## Immediate coverage gate

Before another GPU run:

1. Parse raw MTRAG tasks without using vLLM's lossy single-request loaders.
2. Group and order tasks by conversation and turn.
3. Render each adjacent source/current pair with deterministic RAG formatting.
4. Tokenize with the experiment model and locate exact non-prefix blocks at
   vLLM block size 16.
5. Report candidates by conversation, overlap count, prompt length, and block
   position.
6. Select a small stratified pilot only after natural coverage is known.

Artificial block alignment may be evaluated later as a clearly labelled
controlled ablation. It must not replace the natural-format coverage result.

