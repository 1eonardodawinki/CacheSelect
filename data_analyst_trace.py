# Data Analyst periodic-agent trace generator.
#
# Reimplements the "Data Analyst" agent from Norgren (arXiv:2605.26289,
# Section 4.1) -- tools: query_db, create_chart, export_data, run_pipeline,
# compute_stats -- restructured for periodic reactivation, same as
# travel_planner_trace.py and code_reviewer_trace.py.
#
# Hybrid real/synthetic design: unlike the other two agents, this one's
# FIXED reference material (the part that's identical across every
# activation, analogous to a system prompt) is a REAL excerpt from Apple's
# fiscal 2004 10-K filing, taken verbatim from the FinQA dataset (Chen et
# al., EMNLP 2021, arXiv:2109.00122, github.com/czyssrs/FinQA) -- example
# id "AAPL/2004/page_36.pdf-2". FinQA itself is a static one-document-one-
# question benchmark with no time-series structure, so it can't drive the
# periodic/sliding-window part of this agent on its own. Instead: the real
# filing is the large, realistic, unchanging reference context every
# activation includes (testing whether DeltaCache holds up when the STABLE
# portion is a genuine real document, not just a short instruction block),
# and the periodic monitoring layer on top is synthetic -- a random walk
# seeded from Apple's real 2004 values, continuing the same three metrics
# (net sales, cost of sales, gross margin %) forward in simulated time,
# with one seeded anomaly for ground truth. Same mechanism as
# travel_planner_trace.py's price walk, just applied to financial metrics
# instead of flight prices.

import numpy as np

from trace_common import Activation, ActivationGroundTruth, TraceBundle

# Verbatim from FinQA dev.json, example id "AAPL/2004/page_36.pdf-2"
# (github.com/czyssrs/FinQA, dataset/dev.json). Real filing text, not
# generated or paraphrased.
FINQA_PRE_TEXT = (
    "Net sales of the retail segment grew to $1.185 billion during 2004 from "
    "$621 million and $283 million, in 2003 and 2002, respectively. The "
    "increases in net sales during both 2004 and 2003 reflect the impact of "
    "new store openings for each fiscal year, including the opening of 21 "
    "new stores in 2004 and 25 new stores in 2003. Gross margin for the "
    "three fiscal years ended September 25, 2004 are as follows "
    "(in millions, except gross margin percentages):"
)

FINQA_TABLE = (
    "                          2004      2003      2002\n"
    "net sales               $8279     $6207     $5742\n"
    "cost of sales             6020      4499      4139\n"
    "gross margin             $2259     $1708     $1603\n"
    "gross margin percentage   27.3%     27.5%     27.9%"
)

FINQA_POST_TEXT = (
    "Gross margin declined in fiscal 2004 to 27.3% of net sales from 27.5% "
    "of net sales in 2003. The company's gross margin during fiscal 2004 "
    "declined due to an increase in mix towards lower margin iPod and iBook "
    "sales, pricing actions on certain Power Macintosh G5 models that were "
    "transitioned during the beginning of 2004, higher warranty costs on "
    "certain portable Macintosh products, and higher freight and duty costs "
    "during fiscal 2004."
)

FINQA_REFERENCE = (
    "--- Reference filing excerpt: Apple Inc., FY2004 10-K "
    "(source: FinQA dataset, github.com/czyssrs/FinQA, id AAPL/2004/page_36.pdf-2) ---\n"
    f"{FINQA_PRE_TEXT}\n\n{FINQA_TABLE}\n\n{FINQA_POST_TEXT}\n"
    "--- end reference excerpt ---"
)

# The three tracked metrics, seeded from Apple's real FY2004 values in the
# reference filing above ($ millions, except the percentage).
DEFAULT_METRICS = ["net_sales", "cost_of_sales", "gross_margin_pct"]
REAL_2004_VALUES = {
    "net_sales": 8279.0,
    "cost_of_sales": 6020.0,
    "gross_margin_pct": 27.3,
}

TOOLS_DESCRIPTION = (
    "Tools: query_db(metric), create_chart(metric), export_data(metric), "
    "run_pipeline(), compute_stats(metric)."
)


def build_system_prompt(metrics, change_threshold_pct=15.0):
    # Builds the fixed instruction block that's identical across every
    # activation (the part that should be near-perfectly reusable by prefix
    # caching), plus the embedded real FinQA reference filing. Ends with a
    # strict, rigid output format ("ALERT: ..." / "STATUS: nominal") so
    # scoring in run_baseline.py can just check for an exact substring.
    metrics_str = ", ".join(metrics)
    return (
        "You are a financial data analyst monitoring Apple Inc.'s key "
        f"metrics ({metrics_str}), continuing forward from the company's "
        "FY2004 10-K filing below. You run a periodic pipeline that "
        "recomputes each metric from updated data feeds and alert when a "
        f"tracked metric moves sharply. {TOOLS_DESCRIPTION}\n\n"
        f"{FINQA_REFERENCE}\n\n"
        "Assess risk using only the most recent pipeline check for each "
        "metric, compared to the check before it -- do not consider older "
        "history. "
        f"If any metric's latest value has moved by more than {change_threshold_pct:.0f}% "
        "since the prior check, output exactly one line: "
        "ALERT: <METRIC> <reason>. If no metric qualifies, output exactly: STATUS: nominal."
    )


def _draw_walk(rng, num_checks, drift=0.0, volatility=0.03):
    # Draws one random % change per pipeline check (a "random walk"), so
    # metric values wiggle realistically instead of moving in a straight line.
    return rng.normal(drift, volatility, num_checks)


def _inject_anomaly_return(returns, check_index, pct_move):
    # Takes an array of normal, random % changes and overwrites exactly one
    # of them with our deliberately large, planted move. This is the "answer
    # key" mechanism: we know exactly which check has the anomaly because
    # we're the ones who put it there. Returns a copy so the caller's
    # original array is untouched.
    out = returns.copy()
    out[check_index] = pct_move
    return out


def generate_metric_series(metric, num_checks, start_value, seed,
                            drift=0.0, volatility=0.03,
                            anomaly_check_index=None, anomaly_pct_move=None):
    # Same mechanism as travel_planner_trace.py's generate_price_series:
    # a seeded random walk of % changes, compounded onto the real starting
    # value, with one position optionally overwritten as the planted anomaly.
    rng = np.random.default_rng(seed)
    returns = _draw_walk(rng, num_checks, drift, volatility)
    if anomaly_check_index is not None and anomaly_pct_move is not None:
        returns = _inject_anomaly_return(returns, anomaly_check_index, anomaly_pct_move)
    values = start_value * np.cumprod(1 + returns)
    # gross_margin_pct is a percentage -- clamp to a sane range so it never
    # drifts somewhere nonsensical like negative or above 100 over many steps.
    if metric == "gross_margin_pct":
        values = np.clip(values, 0.0, 100.0)
    return values.tolist()


def format_metric_window(metric, values, window_start_idx):
    # Renders one metric's visible window of pipeline checks as plain text
    # for the prompt, e.g. "net_sales:\n  check 5: 8,412.30$M". `window_start_idx`
    # is added on so the check numbers shown reflect their true position in
    # the full metric history, not just their position within this window.
    unit = "%" if metric == "gross_margin_pct" else "$M"
    lines = [f"{metric}:"]
    for offset, value in enumerate(values):
        check_idx = window_start_idx + offset
        lines.append(f"  check {check_idx}: {value:,.2f}{unit}")
    return "\n".join(lines)


def build_activation_messages(system_prompt, metric_windows, window_start_idx):
    # Assembles the two-message chat payload for one activation: the fixed
    # system prompt (built once, reused every activation, includes the real
    # FinQA excerpt) plus a user message listing every tracked metric's
    # current pipeline-check window.
    blocks = [
        format_metric_window(metric, values, window_start_idx)
        for metric, values in metric_windows.items()
    ]
    user_content = (
        "Latest pipeline checks:\n\n" + "\n\n".join(blocks) +
        "\n\nReview the latest check for each metric and report any threshold breaches."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def generate_trace(
    routes=None,  # accepted for signature compatibility with run_baseline.py's generic call;
                  # "routes" here means "metrics" -- see DEFAULT_METRICS
    num_activations=20,
    window_size=30,
    stride=1,
    seed=42,
    anomaly_activation_index=None,
    anomaly_route=None,  # same compatibility note -- this is the anomaly metric
    anomaly_pct_move=-0.30,
    volatility=0.03,
    start_prices=None,  # same compatibility note -- starting values per metric
    price_drop_threshold_pct=15.0,
):
    # Pure function, same determinism guarantee as travel_planner_trace.py:
    # identical arguments always produce byte-identical output.
    metrics = list(routes) if routes else list(DEFAULT_METRICS)
    if anomaly_activation_index is None:
        anomaly_activation_index = num_activations // 2  # default: plant it roughly in the middle
    if anomaly_route is None:
        anomaly_route = metrics[1] if len(metrics) > 1 else metrics[0]
    # Real FY2004 values from the FinQA filing unless the caller overrides them.
    start_values = start_prices if start_prices else dict(REAL_2004_VALUES)

    # Total pipeline-check history needed to cover every activation's window.
    # E.g. window_size=30, stride=1, num_activations=20 -> 49 checks total,
    # since activation 19's window is checks [19, 49).
    num_checks = window_size + stride * (num_activations - 1)

    # Convert "the anomaly should be the NEWEST check visible in activation N's
    # window" into an absolute index into the full metric history. This is
    # what makes ground truth unambiguous: by construction, no other
    # activation has this check as its newest one, even though a sliding
    # window means this check is still technically visible in several
    # activations before and after.
    anomaly_check_index = window_size - 1 + anomaly_activation_index * stride

    system_prompt = build_system_prompt(metrics, price_drop_threshold_pct)

    # Generate one full value history per metric, continuing forward from its
    # real FY2004 starting value. Only `anomaly_route` gets the anomaly
    # injected; the other metrics are just normal random walks.
    metric_series = {}
    for i, metric in enumerate(metrics):
        is_anomaly_metric = metric == anomaly_route
        metric_series[metric] = generate_metric_series(
            metric=metric,
            num_checks=num_checks,
            start_value=start_values[metric],
            seed=seed + i,  # different-but-deterministic seed per metric, so
                             # metrics don't all move in lockstep
            volatility=volatility,
            anomaly_check_index=anomaly_check_index if is_anomaly_metric else None,
            anomaly_pct_move=anomaly_pct_move if is_anomaly_metric else None,
        )

    # Slice out each activation's window and build its prompt + ground truth.
    activations = []
    for act_idx in range(num_activations):
        window_start = act_idx * stride
        window_end = window_start + window_size
        metric_windows = {m: metric_series[m][window_start:window_end] for m in metrics}
        messages = build_activation_messages(system_prompt, metric_windows, window_start)
        # Flat string version, purely for human-readable dumps (see trace_common.py).
        prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]

        # True on exactly one activation -- the one whose newest check is the
        # planted anomaly. Every other activation should produce "STATUS: nominal".
        expect_flag = act_idx == anomaly_activation_index
        ground_truth = ActivationGroundTruth(
            activation_index=act_idx,
            expect_flag=expect_flag,
            anomaly_entity=anomaly_route if expect_flag else None,
            anomaly_bar_index=anomaly_check_index if expect_flag else None,
            anomaly_magnitude=anomaly_pct_move if expect_flag else None,
        )
        activations.append(Activation(index=act_idx, prompt=prompt, messages=messages, ground_truth=ground_truth))

    return TraceBundle(
        agent_name="data_analyst",
        tickers_or_entities=metrics,
        num_activations=num_activations,
        window_size=window_size,
        stride=stride,
        seed=seed,
        anomaly_activation_index=anomaly_activation_index,
        anomaly_entity=anomaly_route,
        anomaly_magnitude=anomaly_pct_move,
        # Every argument used to build this trace, saved for reproducibility
        # -- so a saved trace file fully documents how to regenerate it.
        generation_params=dict(
            metrics=metrics, num_activations=num_activations, window_size=window_size,
            stride=stride, seed=seed, anomaly_activation_index=anomaly_activation_index,
            anomaly_metric=anomaly_route, anomaly_pct_move=anomaly_pct_move,
            volatility=volatility, start_values=start_values,
            change_threshold_pct=price_drop_threshold_pct,
            finqa_source_id="AAPL/2004/page_36.pdf-2",
        ),
        activations=activations,
    )
