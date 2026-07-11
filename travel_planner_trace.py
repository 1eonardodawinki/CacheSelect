"""Travel Planner periodic-agent trace generator.

Reimplements the "Travel Planner" agent from Norgren (arXiv:2605.26289,
Section 4.1) -- tools: get_weather, search_flights, book_hotel,
search_restaurants, create_itinerary -- but restructured for DeltaCache's
actual target scenario: periodic re-checking of tracked flight prices on a
schedule, rather than Norgren's single 5-turn booking session. Norgren's
version tests within-run monotonic growth (already handled by exact prefix
caching); this version tests cross-activation reuse under a sliding window
of price-check history, where old checks age out and new ones append --
mirroring the design of the (now-superseded) synthetic_trace.py / OHLCV
generator, just with flight prices instead of log lines / stock ticks.

Ground truth follows the same "most recent check only" resolution rule used
in the OHLCV design: the agent is instructed to assess risk based only on
the latest price check relative to the prior one, so ground truth is
unambiguous even though a shocked price lingers inside several consecutive
sliding windows.
"""

import dataclasses

import numpy as np

from trace_common import Activation, ActivationGroundTruth, TraceBundle

# The three routes we pretend the user is watching. Arbitrary but realistic-looking.
DEFAULT_ROUTES = ["LHR-JFK", "LHR-SFO", "LHR-NRT"]

# Copied verbatim (in spirit) from Norgren's Travel Planner tool list -- we
# never actually call these tools, we just describe them in the system
# prompt so the prompt shape/size matches a real tool-calling agent.
TOOLS_DESCRIPTION = (
    "Tools: get_weather(location, date), search_flights(origin, destination, date), "
    "book_hotel(hotel_id, dates), search_restaurants(location), "
    "create_itinerary(components)."
)


def build_system_prompt(routes, price_drop_threshold_pct=15.0):
    """Builds the fixed instruction block that's identical across every
    activation (this is the part that should be near-perfectly reusable by
    prefix caching -- it never changes). Ends with a strict, rigid output
    format ("ALERT: ..." / "STATUS: nominal") rather than open-ended prose,
    because that's what makes automatic scoring in run_baseline.py reliable
    -- we can just check for an exact substring instead of trying to parse
    free-form natural language."""
    routes_str = ", ".join(routes)
    return (
        "You are a travel monitoring agent. You track flight prices for a set of "
        f"routes ({routes_str}) and alert the user when a tracked route's price "
        "drops significantly, so they can book while it's cheap. "
        f"{TOOLS_DESCRIPTION} "
        "Assess risk using only the most recent price check for each route, "
        "compared to the check before it -- do not consider older history. "
        f"If any route's latest price has dropped by more than {price_drop_threshold_pct:.0f}% "
        "since the prior check, output exactly one line: "
        "ALERT: <ROUTE> <reason>. If no route qualifies, output exactly: STATUS: nominal."
    )


def _draw_price_walk(rng, num_checks, start_price, drift=0.0, volatility=0.03):
    """Draws one random % change per price check (a "random walk"), so
    prices wiggle realistically instead of moving in a straight line.
    `start_price` is unused here (kept for signature symmetry/clarity with
    generate_price_series) -- the actual starting value is applied later."""
    returns = rng.normal(drift, volatility, num_checks)
    return returns


def _inject_anomaly_return(returns, check_index, pct_move):
    """Takes an array of normal, random price-change values and overwrites
    exactly one of them with our deliberately large, planted move (e.g.
    -0.30 for a 30% crash). This is the "answer key" mechanism: we know
    exactly which check has the anomaly because we're the ones who put it
    there. Returns a copy so the caller's original array is untouched."""
    out = returns.copy()
    out[check_index] = pct_move
    return out


def generate_price_series(route, num_checks, start_price, seed,
                           drift=0.0, volatility=0.03,
                           anomaly_check_index=None, anomaly_pct_move=None):
    """Builds one route's full price history as a list of dollar amounts.
    Uses a fixed `seed` so the exact same numbers come out every time this
    is called with the same arguments -- critical for the --apc on and
    --apc off runs to see identical prices, not just identically-structured
    ones."""
    rng = np.random.default_rng(seed)
    returns = _draw_price_walk(rng, num_checks, start_price, drift, volatility)
    if anomaly_check_index is not None and anomaly_pct_move is not None:
        returns = _inject_anomaly_return(returns, anomaly_check_index, anomaly_pct_move)
    # Turn a list of % changes into a list of actual prices by compounding
    # them onto the starting price, e.g. [100, +2%, -1%] -> [100, 102, 100.98].
    prices = start_price * np.cumprod(1 + returns)
    return prices.tolist()


def format_price_window(route, prices, window_start_idx):
    """Renders one route's visible window of price checks as plain text for
    the prompt, e.g.:
        LHR-JFK:
          check 5: $512.30
          check 6: $498.10
    `window_start_idx` is added on so the check numbers shown in the prompt
    reflect their true position in the full price history, not just their
    position within this particular window."""
    lines = [f"{route}:"]
    for offset, price in enumerate(prices):
        check_idx = window_start_idx + offset
        lines.append(f"  check {check_idx}: ${price:,.2f}")
    return "\n".join(lines)


def build_activation_messages(system_prompt, route_windows, window_start_idx):
    """Assembles the two-message chat payload for one activation: the fixed
    system prompt (built once, reused every activation) plus a user message
    listing every tracked route's current price window."""
    blocks = [
        format_price_window(route, prices, window_start_idx)
        for route, prices in route_windows.items()
    ]
    user_content = (
        "Latest price checks:\n\n" + "\n\n".join(blocks) +
        "\n\nReview the latest check for each route and report any threshold breaches."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def generate_trace(
    routes=None,
    num_activations=20,
    window_size=30,
    stride=1,
    seed=42,
    anomaly_activation_index=None,
    anomaly_route=None,
    anomaly_pct_move=-0.30,
    volatility=0.03,
    start_prices=None,
    price_drop_threshold_pct=15.0,
):
    """Builds a complete TraceBundle: a sequence of `num_activations`
    activations, each showing a `window_size`-check-wide slice of price
    history that slides forward by `stride` checks every activation.

    IMPORTANT: this is a pure function -- calling it twice with the same
    arguments always produces byte-identical output (same seed, no
    wall-clock/random state leaking in from outside). That determinism is
    what lets --apc on and --apc off be compared fairly.
    """
    routes = list(routes) if routes else list(DEFAULT_ROUTES)
    if anomaly_activation_index is None:
        anomaly_activation_index = num_activations // 2  # default: plant it roughly in the middle
    if anomaly_route is None:
        anomaly_route = routes[1] if len(routes) > 1 else routes[0]
    if start_prices is None:
        # Just spread out starting prices so routes are visually distinguishable.
        start_prices = {r: 500.0 + 100.0 * i for i, r in enumerate(routes)}

    # Total price-check history needed to cover every activation's window.
    # E.g. window_size=30, stride=1, num_activations=20 -> 49 checks total,
    # since activation 19's window is checks [19, 49).
    num_checks = window_size + stride * (num_activations - 1)

    # Convert "the anomaly should be the NEWEST bar visible in activation N's
    # window" into an absolute index into the full price-check history. This
    # is what makes ground truth unambiguous: by construction, no other
    # activation has this bar as its newest one, even though a sliding
    # window means this bar is still technically visible in several
    # activations before and after.
    anomaly_check_index = window_size - 1 + anomaly_activation_index * stride

    system_prompt = build_system_prompt(routes, price_drop_threshold_pct)

    # Generate one full price history per route. Only `anomaly_route` gets
    # the anomaly injected; the other routes are just normal random walks.
    price_series = {}
    for i, route in enumerate(routes):
        is_anomaly_route = route == anomaly_route
        price_series[route] = generate_price_series(
            route=route,
            num_checks=num_checks,
            start_price=start_prices[route],
            seed=seed + i,  # different-but-deterministic seed per route, so
                             # routes don't all move in lockstep. Using
                             # seed+i (not hash(route)) keeps this
                             # reproducible across Python runs/versions.
            volatility=volatility,
            anomaly_check_index=anomaly_check_index if is_anomaly_route else None,
            anomaly_pct_move=anomaly_pct_move if is_anomaly_route else None,
        )

    # Slice out each activation's window and build its prompt + ground truth.
    activations = []
    for act_idx in range(num_activations):
        window_start = act_idx * stride
        window_end = window_start + window_size
        route_windows = {r: price_series[r][window_start:window_end] for r in routes}
        messages = build_activation_messages(system_prompt, route_windows, window_start)
        # Flat string version, purely for human-readable dumps (see trace_common.py).
        prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]

        # True on exactly one activation -- the one whose newest bar is the
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
        agent_name="travel_planner",
        tickers_or_entities=routes,
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
            routes=routes, num_activations=num_activations, window_size=window_size,
            stride=stride, seed=seed, anomaly_activation_index=anomaly_activation_index,
            anomaly_route=anomaly_route, anomaly_pct_move=anomaly_pct_move,
            volatility=volatility, start_prices=start_prices,
            price_drop_threshold_pct=price_drop_threshold_pct,
        ),
        activations=activations,
    )


if __name__ == "__main__":
    # Quick manual sanity check: print a tiny 5-activation trace so a human
    # can eyeball that the prompts look right, without needing vLLM or a GPU.
    bundle = generate_trace(num_activations=5, window_size=10, stride=1, seed=1)
    for a in bundle.activations:
        print(f"=== activation {a.index} (expect_flag={a.ground_truth.expect_flag}) ===")
        print(a.prompt)
        print()
