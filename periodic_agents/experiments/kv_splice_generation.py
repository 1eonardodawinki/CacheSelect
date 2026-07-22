# Experiment 2: does splicing a STALE (pre-drop) K,V for one check into an
# otherwise-correct forward pass actually change the model's generated
# answer -- not just its internal representation? Experiment 1
# (kv_contamination.py) measured representational distance; this measures
# the behavioural consequence, which is the thing that actually determines
# whether approximate reuse is safe. A large vector distance that never
# flips an answer is a non-issue; a small one that does is not.
#
# Setup: same two Travel Planner activations as Experiment 1 (t = checks
# 0..5, t+1 = checks 1..6), but this time with a REAL planted anomaly at
# check 6 -- the newest check in t+1, on FOCUS_ROUTE -- so there's a
# genuine ground-truth "ALERT" answer to test whether contamination
# disrupts it. Checks 1..5 are explicitly, by the system prompt's own
# rule, NOT supposed to matter to that decision ("assess risk using only
# the most recent check ... do not consider older history"); this
# experiment asks whether that rule actually holds up once one of those
# "irrelevant" checks is corrupted.
#
# For each shared check k in 1..5: run activation t+1's prompt normally up
# to the end of check k's tokens, splice check k's K,V (every layer) for
# the STALE version computed as part of activation t instead, continue the
# forward pass over the rest of the prompt (checks k+1..6 and the
# generation-prompt suffix) using the spliced cache -- exactly what
# content-only block hashing would hand the model in a real reuse -- then
# greedily decode the model's actual answer. Compared against the TRUE
# (unspliced) baseline answer and the ground truth.
#
# Runs entirely locally via plain transformers (CPU/MPS), no vLLM or GPU
# cluster. Usage (from the repo root):
#     python -m periodic_agents.experiments.kv_splice_generation

import csv
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from periodic_agents.agents.travel_planner_trace import DEFAULT_ROUTES, generate_trace
from periodic_agents.experiments.kv_contamination import (
    char_span_to_token_indices,
    extract_layer_kv,
    find_check_spans,
    find_route_block,
    pick_device,
)
from periodic_agents.harness.run_baseline import score_response

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FOCUS_ROUTE = DEFAULT_ROUTES[0]  # same route as Experiment 1; also given the anomaly here
MAX_NEW_TOKENS = 24
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "experiment2_kv_splice_generation.csv")


def render_and_tokenize(tokenizer, messages):
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, return_offsets_mapping=True, add_special_tokens=False)
    return rendered, encoded["input_ids"], encoded["offset_mapping"]


def get_check_token_spans(tokenizer, messages, route):
    # Same offset-mapping approach as Experiment 1 -- see kv_contamination.py
    # for why the trailing newline is deliberately excluded from each span.
    rendered, input_ids, offset_mapping = render_and_tokenize(tokenizer, messages)
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    user_offset = rendered.find(user_content)
    if user_offset == -1:
        raise ValueError("Chat template altered the user content unexpectedly.")
    block_start, block_text = find_route_block(user_content, route)
    check_char_spans = find_check_spans(block_text, block_start + user_offset)
    check_token_spans = {
        idx: char_span_to_token_indices(offset_mapping, s, e)
        for idx, (s, e) in check_char_spans.items()
    }
    return input_ids, check_token_spans


def splice_cache_positions(cache, num_layers, splice_positions, stale_cache, stale_positions):
    # Overwrites, in place, every layer's K,V at `splice_positions` (indices
    # into `cache`'s own sequence dimension) with the corresponding K,V
    # taken from `stale_cache` at `stale_positions` (indices into
    # `stale_cache`'s sequence dimension -- a different absolute position,
    # since the check sits elsewhere in the two activations' windows).
    assert len(splice_positions) == len(stale_positions), (
        "Check token count differs between the two activations -- can't splice "
        "position-for-position. Likely a formatting difference (e.g. digit count)."
    )
    for layer_idx in range(num_layers):
        key, value = extract_layer_kv(cache, layer_idx)
        stale_key, stale_value = extract_layer_kv(stale_cache, layer_idx)
        for dst, src in zip(splice_positions, stale_positions):
            key[:, :, dst, :] = stale_key[:, :, src, :]
            value[:, :, dst, :] = stale_value[:, :, src, :]


def greedy_decode(model, tokenizer, cache, last_logits, device, max_new_tokens=MAX_NEW_TOKENS):
    # Continues generation from an already-populated cache. `last_logits` is
    # the model's output at the final already-processed position (i.e. the
    # prediction for the first new token); every step after that re-invokes
    # the model on just the one new token, extending `cache` in place.
    generated_ids = []
    logits = last_logits
    for _ in range(max_new_tokens):
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break
        generated_ids.append(next_id)
        with torch.no_grad():
            out = model(torch.tensor([[next_id]], device=device), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_generation(model, tokenizer, device, input_ids):
    # Plain, unmodified generation -- the "everything computed correctly"
    # baseline for one activation's full prompt.
    with torch.no_grad():
        out = model(torch.tensor([input_ids], device=device), use_cache=True)
    return greedy_decode(model, tokenizer, out.past_key_values, out.logits, device)


def run_spliced_generation(model, tokenizer, device, input_ids_t1, spans_t1, check_idx,
                            cache_stale, spans_stale):
    # The actual splice: process t+1's prompt up to and including check
    # `check_idx`, overwrite that check's K,V with the stale version from
    # `cache_stale`, then continue processing the remainder of the prompt
    # (later checks + the generation-prompt suffix) attending through the
    # now-stale block -- exactly what a content-only cache hit would hand
    # the model -- before decoding the answer.
    check_positions = spans_t1[check_idx]
    prefix_ids = input_ids_t1[: check_positions[-1] + 1]
    suffix_ids = input_ids_t1[check_positions[-1] + 1 :]

    with torch.no_grad():
        prefix_out = model(torch.tensor([prefix_ids], device=device), use_cache=True)
    cache = prefix_out.past_key_values
    num_layers = model.config.num_hidden_layers

    # check_idx's tokens are the LAST len(check_positions) positions of the
    # prefix we just processed -- splice those, pulling the stale values
    # from cache_stale's own copy of the same check.
    prefix_len = len(prefix_ids)
    splice_positions = list(range(prefix_len - len(check_positions), prefix_len))
    stale_positions = spans_stale[check_idx]
    splice_cache_positions(cache, num_layers, splice_positions, cache_stale, stale_positions)

    with torch.no_grad():
        suffix_out = model(torch.tensor([suffix_ids], device=device), past_key_values=cache, use_cache=True)
    return greedy_decode(model, tokenizer, suffix_out.past_key_values, suffix_out.logits, device)


def main():
    device = pick_device()
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME} (tokenizer + weights) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # bf16, not fp16 -- matches the real GPU baseline run's precision
    # (vLLM ran this model in bf16). Precision differences can genuinely
    # flip a close-call greedy argmax, so this keeps the comparison fair.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    # Three separate real-model runs (window_size=6, then 6 with a -0.60
    # drop, then 15 with bf16) all produced the SAME baseline output --
    # "STATUS: nominal" -- regardless of how obvious the anomaly was. That's
    # not noise, it's a real, pre-existing model limitation: the real GPU
    # baseline run in the experiment log missed its one genuine anomaly the
    # exact same way. Chasing a config where this 1.5B model reliably
    # detects a real anomaly is a losing strategy -- there's already strong
    # evidence it won't.
    #
    # Flipping the test instead: rather than checking whether splicing
    # causes a MISSED alert (needs a baseline that correctly alerts --
    # unreliable), check whether splicing causes a FALSE alert in a case
    # with no anomaly anywhere (a baseline the model reliably gets right,
    # given its demonstrated bias toward "nominal"). Equally valid evidence
    # about contamination safety, just testing the other failure direction.
    #
    # num_activations=3 with the anomaly pinned to activation index 2 keeps
    # it completely outside both t (index 0, checks 0..14) and t+1 (index 1,
    # checks 1..15)'s visible windows -- neither activation used for testing
    # ever sees the anomaly, so t+1's "STATUS: nominal" ground truth is
    # genuinely, unambiguously correct.
    trace = generate_trace(
        num_activations=3, window_size=15, stride=1, seed=1,
        anomaly_activation_index=2, anomaly_route=FOCUS_ROUTE,
        anomaly_pct_move=-0.60,
    )
    activation_t, activation_t1 = trace.activations[0], trace.activations[1]
    ground_truth = activation_t1.ground_truth
    entities = trace.tickers_or_entities
    print(f"Ground truth for t+1: expect_flag={ground_truth.expect_flag}, "
          f"anomaly_entity={ground_truth.anomaly_entity}")

    input_ids_t, spans_t = get_check_token_spans(tokenizer, activation_t.messages, FOCUS_ROUTE)
    input_ids_t1, spans_t1 = get_check_token_spans(tokenizer, activation_t1.messages, FOCUS_ROUTE)
    shared_checks = sorted(set(spans_t) & set(spans_t1))
    print(f"Shared checks: {shared_checks}")

    with torch.no_grad():
        cache_t_out = model(torch.tensor([input_ids_t], device=device), use_cache=True)
    cache_t = cache_t_out.past_key_values

    print("Running baseline (unspliced) generation for t+1 ...")
    baseline_text = run_generation(model, tokenizer, device, input_ids_t1)
    baseline_correct, baseline_flagged = score_response(ground_truth, baseline_text, entities)
    print(f"Baseline output: {baseline_text!r} -> correct={baseline_correct}")

    if not baseline_correct:
        print(
            "\nWARNING: the model got the UNSPLICED baseline wrong -- there is no correct "
            "answer for splicing to break, so any 'flips' result below is not meaningful. "
            "This is a model-capability failure (same failure mode seen on the real GPU "
            "baseline runs), not evidence about contamination either way. Try a larger "
            "anomaly_pct_move, a different seed, or a larger window_size before trusting "
            "the results.\n"
        )

    rows = []
    for check_idx in shared_checks:
        spliced_text = run_spliced_generation(
            model, tokenizer, device, input_ids_t1, spans_t1, check_idx, cache_t, spans_t
        )
        spliced_correct, spliced_flagged = score_response(ground_truth, spliced_text, entities)
        answer_changed = spliced_text.strip() != baseline_text.strip()
        print(f"check {check_idx} (distance {check_idx}): spliced output: {spliced_text!r} "
              f"-> correct={spliced_correct}, changed={answer_changed}")
        rows.append({
            "check_index": check_idx,
            "distance_from_drop": check_idx,
            "baseline_output": baseline_text,
            "baseline_correct": baseline_correct,
            "spliced_output": spliced_text,
            "spliced_correct": spliced_correct,
            "answer_changed": answer_changed,
        })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")

    flips = sum(1 for r in rows if not r["spliced_correct"] and r["baseline_correct"])
    print(f"\n{flips}/{len(rows)} spliced checks flipped a correct baseline answer to incorrect.")


if __name__ == "__main__":
    main()
