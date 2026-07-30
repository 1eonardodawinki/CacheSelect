# LLM Request Ledger

Every conceptual LLM generation request made by CacheSelect experiments must
be recorded with `RequestRecorder`.

A conceptual request is one externally meaningful generation attempt, such as
a full-computation baseline, an exact-cache request, an approximate repair, a
control, or a warm-up. Individual model forwards used to build KV state,
process a suffix, or decode one token are internal steps of that request and
are not separate ledger entries.

## Event format

The recorder writes append-only JSONL:

1. `request_started` is flushed before inference and contains the complete
   messages or rendered prompt, token IDs when available, model, backend,
   sampling configuration, policy metadata, and evaluation metadata.
2. `request_completed` contains the full generated text, output token IDs when
   available, metrics, and scoring results.
3. `request_failed` replaces the completion event when inference raises and
   contains the exception and traceback.

The two events share `request_id`, `run_id`, and `invocation_id`.

Ground truth is stored under `evaluation`, never under `model_input`. In the
future planner path it must also remain unavailable to feature extraction and
policy selection.

## Storage

By default logs are written to:

```text
cacheselect/request_logs/<run-id>.jsonl
```

Set `CACHESELECT_REQUEST_LOG_DIR` to place them on durable experiment storage.
The default directory is ignored by Git because prompts may contain sensitive
or licensed data. Selected, reviewed ledgers can be copied into a release
artefact deliberately.

## Required usage

Create one recorder per process invocation:

```python
recorder = RequestRecorder(
    run_id="rag-musique-baseline",
    model=model_name,
    backend="vllm",
)
```

Persist the input before calling the model:

```python
pending = recorder.start(
    model_input={"messages": messages, "rendered_prompt": rendered_prompt},
    sampling={"temperature": 0.0},
    metadata={"policy": "FULL"},
    evaluation={"ground_truth": answer},
)
```

Then complete or fail the same handle:

```python
try:
    output = call_model(...)
except BaseException as error:
    pending.fail(error)
    raise
else:
    pending.complete(output={"text": output})
```

New LLM entry points are incomplete until they follow this contract.

## Completeness audit

After an experiment, verify that every started request has exactly one
completion or failure:

```bash
python -m observability.validate_ledger request_logs/<run-id>.jsonl
```

The command exits non-zero for an interrupted request. A recorded failure is a
complete ledger entry, but should still be handled according to the experiment
protocol.
