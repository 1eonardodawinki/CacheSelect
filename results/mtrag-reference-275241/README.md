# MTRAG natural reference run 275241

This A16 run computed 21 training and six validation answers from scratch with
Qwen2.5-1.5B-Instruct. Both append-only request ledgers are complete: 27 started,
27 completed, zero failed, and zero incomplete requests. The sealed test
manifest was not opened.

## Blinded correctness review

`manual-review-blinded.json` contains only opaque review IDs, model answers,
expected answers, and lexical diagnostics. Split, collection, and task identity
are stored separately in `manual-review-key.json`. All verdicts were frozen in
`manual-verdicts-blinded.json` before the identity key was inspected.

| Split | Pass | Concern | Fail | Total |
|---|---:|---:|---:|---:|
| Train | 3 | 7 | 11 | 21 |
| Validation | 3 | 0 | 3 | 6 |
| Total | 6 | 7 | 14 | 27 |

Only a `pass` reference is currently eligible to create semantic causal labels.
The broader sample therefore demonstrates that the 1.5B model is unreliable on
many MTRAG questions; it does not provide enough approved natural transitions
to train the three selectors as planned. Concern and fail rows remain useful as
auditable workload evidence but must not silently become REUSE labels.

## SHA-256

- Metadata: `f325ff415e654e402231b6a3c4ee34427cc9b6d00c2b2ef5f9ad945e66113945`
- Training references: `d2988bfad3ea00b8e44c221011b3eaefd3a2c6f68e240fee9047c54d825f60b9`
- Validation references: `c275a4f002c63250e6a86633c85d85ac098ab7a682f8cc1e2d21500d255c6f86`
- Blinded review: `6ce29f8df5391355be839d5781120c0d9d15b3ec4bc6f8ee7d9d7cef55cafa2a`
- Identity key: `32c1459b830d1a4af720f8a3ca95fc7ebb6c750450042d2392884871ce2509b0`
- Frozen verdicts: `255e5d733aaa27cdac7b9219d19945c9476139c4d67a4b5e93576be8a77ff9ce`

The full JSONL request ledgers are retained locally under `request-logs/` but
are intentionally not committed because they duplicate large prompt payloads.
