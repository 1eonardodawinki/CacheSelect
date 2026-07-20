# Baseline harness: runs a periodic-agent trace against unmodified vLLM and
# measures Time-To-First-Token (TTFT) and accuracy, once per APC configuration.
#
# Agent-agnostic -- works against any trace generator module that exposes a
# generate_trace(...) -> TraceBundle function using the shared dataclasses in
# trace_common.py. Supports: travel_planner, code_reviewer, data_analyst.
#
# Requires a CUDA GPU -- this machine does not have one; run on a GPU host.
#
# Usage (run as a module from the repo root, not as a bare script -- that
# keeps the repo root on sys.path so `agents.*` and `harness.*` both resolve).
# Two separate process runs are required since APC is fixed at engine
# construction time -- the second run reloads the exact trace the first run
# saved, guaranteeing both see byte-identical prompts):
#
#     python -m harness.run_baseline --agent travel_planner --apc on  --tag run1 \
#         --save-trace results/trace_run1.json
#     python -m harness.run_baseline --agent travel_planner --apc off --tag run1 \
#         --trace-file results/trace_run1.json

import argparse
import csv
import dataclasses
import json
import os
import statistics
import textwrap

from harness.trace_common import TraceBundle, load_trace, save_trace

# Cache of already-imported agent trace modules, so repeated calls to
# _get_agent_module() during one run don't re-import needlessly.
AGENT_MODULES = {}


def _get_agent_module(agent_name):
    # Looks up which trace-generator module to use for a given --agent
    # value. Nothing else in this file needs to change when a new agent is
    # added, since everything downstream only talks to the shared
    # TraceBundle/Activation shapes from trace_common.py.
    if agent_name not in AGENT_MODULES:
        if agent_name == "travel_planner":
            import agents.travel_planner_trace as mod
        elif agent_name == "code_reviewer":
            import agents.code_reviewer_trace as mod
        elif agent_name == "data_analyst":
            import agents.data_analyst_trace as mod
        else:
            raise ValueError(f"Unknown agent '{agent_name}'.")
        AGENT_MODULES[agent_name] = mod
    return AGENT_MODULES[agent_name]


@dataclasses.dataclass
class ActivationResult:
    # Everything we recorded for one activation's request to the model --
    # one row of the final results table.

    activation_index: int
    apc: str  # "on" or "off" -- which configuration produced this result

    # Token count using the REAL model's tokenizer (not tiktoken) -- this is
    # what actually determines KV-cache block boundaries, so it's the only
    # count that means anything for interpreting num_cached_tokens below.
    prompt_token_count: int

    # How many of those prompt tokens vLLM served from its prefix cache
    # instead of recomputing. This is the direct, mechanistic explanation
    # for any TTFT difference between --apc on and --apc off -- and later,
    # the number DeltaCache is trying to raise.
    num_cached_tokens: int | None

    ttft_seconds: float | None  # time to first token -- the actual latency metric we care about
    output_text: str  # the model's raw generated text, kept for debugging/spot-checks

    # Copied over from the trace's ground truth, so the results file is
    # self-contained -- you don't need the original trace file to see what
    # SHOULD have happened on each row.
    expect_flag: bool
    anomaly_entity: str | None

    predicted_flagged_entities: list  # what the model actually flagged, per score_response()
    correct: bool  # did the model's answer match the ground truth exactly


def score_response(ground_truth, text, entities):
    # Grades one model response against its ground truth. Looks for the
    # exact "ALERT: <ENTITY>" pattern the system prompt instructed the model
    # to use -- deliberately a strict substring check, not fuzzy text
    # understanding, because the system prompt forces a rigid output format
    # specifically so this check can be simple and reliable.
    #
    # Correct means: if an anomaly was planted, the model flagged EXACTLY that
    # one entity and nothing else; if nothing was planted, the model flagged
    # nothing at all. Partial credit (e.g. flagging the right entity plus an
    # extra false alarm) counts as incorrect.
    text_upper = text.upper()
    flagged = [e for e in entities if f"ALERT: {e.upper()}" in text_upper]
    if ground_truth.expect_flag:
        correct = flagged == [ground_truth.anomaly_entity]
    else:
        correct = flagged == []
    return correct, flagged


def load_or_generate_trace(args) -> TraceBundle:
    # Either reloads a previously-saved trace (--trace-file) or generates a
    # fresh one and optionally saves it (--save-trace). Reloading is how the
    # --apc off run guarantees it sees the exact same prompts the --apc on run
    # already saw -- see the module comment's Usage example at the top of this file.
    if args.trace_file:
        return load_trace(args.trace_file)
    mod = _get_agent_module(args.agent)
    bundle = mod.generate_trace(
        num_activations=args.num_activations,
        window_size=args.window_size,
        stride=args.stride,
        seed=args.seed,
        anomaly_activation_index=args.anomaly_activation_index,
    )
    if args.save_trace:
        os.makedirs(os.path.dirname(args.save_trace), exist_ok=True)
        save_trace(bundle, args.save_trace)
    return bundle


def build_llm(args):
    # Imported here, not at module top, so this file can be inspected/linted
    # on a machine without vLLM installed (e.g. this Mac).
    from vllm import LLM

    return LLM(
        model=args.model,
        enable_prefix_caching=(args.apc == "on"),  # the actual on/off switch we're testing
        disable_log_stats=False,  # LLM() defaults this to True, which silently
                                   # disables RequestOutput.metrics (and therefore
                                   # ttft_seconds below) -- must be explicitly overridden.
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
    )


def warmup(llm, sampling_params):
    # Fires one throwaway request before the real sweep starts, to absorb
    # any one-time first-call overhead (e.g. CUDA graph capture) so it doesn't
    # pollute activation 0's timing.
    #
    # The warmup text is deliberately generic and shares no words/tokens with
    # the real system prompt. If it did, under --apc on this warmup call would
    # itself populate the prefix cache with (part of) the real system prompt
    # before activation 0 ever runs -- silently giving activation 0 a "free"
    # cache hit that would never happen in a real cold start, and making the
    # --apc on vs --apc off comparison unfair.
    outputs = llm.chat(
        [[{"role": "user", "content": "Warm up. Reply with one word."}]],
        sampling_params,
        use_tqdm=False,
    )
    assert outputs[0].metrics is not None, (
        "RequestOutput.metrics is None -- disable_log_stats must be False."
    )


def run_sweep(llm, tokenizer, trace: TraceBundle, sampling_params, apc_label):
    # Runs every activation in the trace through the model, ONE AT A TIME,
    # in order -- and records the result of each.
    #
    # Deliberately sequential (a Python for-loop calling llm.chat() once per
    # activation), NOT one batched llm.chat() call with all prompts at once.
    # Batching would let vLLM's scheduler interleave multiple activations'
    # prefill/decode work in the same forward pass, which could inflate or
    # deflate any individual activation's measured TTFT depending on what else
    # happens to be running alongside it -- that's not how a real periodic
    # agent behaves (it checks in once every 15 minutes, not all at once), and
    # it would make the TTFT numbers meaningless for our purposes.
    results = []
    for activation in trace.activations:
        prompt_token_count = len(tokenizer.encode(activation.prompt))
        outputs = llm.chat([activation.messages], sampling_params, use_tqdm=False)
        output = outputs[0]
        assert output.metrics is not None
        text = output.outputs[0].text
        correct, flagged = score_response(
            activation.ground_truth, text, trace.tickers_or_entities
        )
        results.append(ActivationResult(
            activation_index=activation.index,
            apc=apc_label,
            prompt_token_count=prompt_token_count,
            num_cached_tokens=output.num_cached_tokens,
            ttft_seconds=output.metrics.first_token_latency,
            output_text=text,
            expect_flag=activation.ground_truth.expect_flag,
            anomaly_entity=activation.ground_truth.anomaly_entity,
            predicted_flagged_entities=flagged,
            correct=correct,
        ))
    return results


def save_results(results, output_dir, agent, apc_label, tag):
    # Writes the results table to both CSV (easy to eyeball / open in a
    # spreadsheet) and JSON (easy to reload programmatically for plotting
    # later) -- same data, two formats, filenames tagged by agent, run name,
    # and APC setting, so runs for different agents (or on/off pairs) never
    # overwrite each other even if --tag is reused across agents.
    os.makedirs(output_dir, exist_ok=True)
    base = f"{agent}_{tag}_apc-{apc_label}"
    csv_path = os.path.join(output_dir, base + ".csv")
    json_path = os.path.join(output_dir, base + ".json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(ActivationResult)])
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))

    with open(json_path, "w") as f:
        json.dump([dataclasses.asdict(r) for r in results], f, indent=2)

    return csv_path, json_path


def save_transcript(trace: TraceBundle, results, output_dir, agent, apc_label, tag):
    # Writes one human-readable text file per (agent, tag, apc) combination
    # containing the exact system+user prompt sent to the model and its raw
    # output, for every activation in order -- so you can see precisely what
    # went in and what came out without cross-referencing the trace JSON
    # against the results CSV by activation_index yourself.
    #
    # Same naming convention as save_results, so it's naturally overwritten
    # on a rerun with the same --tag (giving you "the last run") and
    # naturally preserved across runs if you vary --tag (giving you "all
    # runs", same as the CSV/JSON already do).
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{agent}_{tag}_apc-{apc_label}_transcript.txt")
    results_by_index = {r.activation_index: r for r in results}

    with open(path, "w") as f:
        for activation in trace.activations:
            r = results_by_index[activation.index]
            system_msg = next(m["content"] for m in activation.messages if m["role"] == "system")
            user_msg = next(m["content"] for m in activation.messages if m["role"] == "user")
            f.write("=" * 80 + "\n")
            f.write(
                f"Activation {activation.index} | expect_flag={r.expect_flag} "
                f"| anomaly_entity={r.anomaly_entity} | correct={r.correct}\n"
            )
            f.write("=" * 80 + "\n\n")
            f.write("--- SYSTEM PROMPT ---\n")
            # build_system_prompt() returns one unbroken paragraph -- the
            # actual model input, which must stay byte-identical for the
            # caching measurement. Wrapping only here, for readability, never
            # touches that string; it's applied purely on the way to disk.
            f.write(textwrap.fill(system_msg, width=100) + "\n\n")
            f.write("--- USER MESSAGE ---\n")
            f.write(user_msg + "\n\n")
            f.write("--- MODEL OUTPUT ---\n")
            f.write(r.output_text + "\n\n")

    return path


def print_summary(results):
    # Prints a quick human-readable summary to the terminal after a run, so
    # you don't have to open the CSV just to sanity-check that something
    # reasonable happened. The cached_tokens/prompt ratio is the single most
    # telling number here: under --apc off it should sit at ~0% for every
    # activation; under --apc on with vanilla (unmodified) vLLM, it should
    # jump up after activation 0 but then stay roughly flat regardless of how
    # much the underlying data actually overlaps -- that flatness is the
    # literal "vanilla caching doesn't handle periodic agents well" result
    # this whole baseline exists to demonstrate, before DeltaCache changes it.
    ttfts = [r.ttft_seconds for r in results if r.ttft_seconds is not None]
    accuracy = sum(1 for r in results if r.correct) / len(results)
    cache_ratios = [
        (r.num_cached_tokens / r.prompt_token_count) if r.num_cached_tokens else 0.0
        for r in results
    ]
    print(f"n_activations       : {len(results)}")
    print(f"accuracy             : {accuracy:.1%}")
    print(f"TTFT mean/median (s) : {statistics.mean(ttfts):.3f} / {statistics.median(ttfts):.3f}")
    print(f"cached_tokens/prompt : mean {statistics.mean(cache_ratios):.1%} "
          f"(per-activation: {[f'{c:.0%}' for c in cache_ratios]})")


def build_arg_parser():
    # Defines every CLI flag this script accepts -- which agent/trace to run,
    # the APC on/off switch, trace generation parameters, model/GPU settings,
    # and output options. See the module comment at the top of this file for
    # example invocations.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True, choices=["travel_planner", "code_reviewer", "data_analyst"])
    p.add_argument("--apc", required=True, choices=["on", "off"])
    p.add_argument("--trace-file", default=None, help="Reload a previously saved trace (for a matched on/off pair).")
    p.add_argument("--save-trace", default=None, help="Save the generated trace to this path.")
    p.add_argument("--num-activations", type=int, default=20)
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--anomaly-activation-index", type=int, default=None)
    # Ungated on HuggingFace (unlike Llama-3.1, which Norgren's own paper
    # used but which requires an approved access request) -- picked so
    # anyone can run this without extra setup friction. Override with
    # --model if you have Llama access and want closer comparability to
    # Norgren's numbers, or a smaller model (e.g. Qwen2.5-1.5B-Instruct) for
    # fast iteration.
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)  # greedy decoding: same input -> same
                                                                 # output every time, so any difference
                                                                 # between apc on/off runs is due to
                                                                 # caching, not random sampling variance
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)  # set explicitly, rather than
                                                                 # letting vLLM auto-derive a much
                                                                 # larger default from the model
                                                                 # config, to avoid over-allocating
                                                                 # KV cache memory for context
                                                                 # lengths this benchmark never uses
    p.add_argument("--dtype", default="auto")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--tag", default="baseline")
    p.add_argument("--smoke-test", action="store_true", help="Run only 2 activations, print raw output, then exit.")
    return p


def main():
    # Entry point: parse args, get/generate the trace, load the model,
    # warm up, run the full sweep, save results, print a summary -- the
    # complete flow for one process invocation (one APC configuration).
    from vllm import SamplingParams

    args = build_arg_parser().parse_args()
    trace = load_or_generate_trace(args)

    if args.smoke_test:
        # Truncate to a tiny trace so a config/API mistake fails in seconds,
        # not after a multi-minute full sweep.
        trace = dataclasses.replace(trace, activations=trace.activations[:2], num_activations=2)

    llm = build_llm(args)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    warmup(llm, sampling_params)
    results = run_sweep(llm, tokenizer, trace, sampling_params, apc_label=args.apc)

    tag = "smoke" if args.smoke_test else args.tag  # smoke-test results never
                                                       # overwrite real run results
    csv_path, json_path = save_results(results, args.output_dir, args.agent, args.apc, tag)
    transcript_path = save_transcript(trace, results, args.output_dir, args.agent, args.apc, tag)
    print(f"Saved {csv_path}, {json_path}, and {transcript_path}")
    print_summary(results)

    if args.smoke_test:
        # Extra verbose dump for smoke tests specifically, so you can
        # eyeball the actual prompt/response pair instead of just trusting
        # the summary numbers.
        for r in results:
            print(f"\n--- activation {r.activation_index} ---")
            print(f"expect_flag={r.expect_flag} anomaly_entity={r.anomaly_entity}")
            print(f"predicted_flagged={r.predicted_flagged_entities} correct={r.correct}")
            print(f"output_text: {r.output_text!r}")


if __name__ == "__main__":
    main()
