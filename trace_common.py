"""Shared trace data structures, used by every agent-specific trace generator
(travel_planner_trace.py, and later code_reviewer_trace.py / data_analyst_trace.py).

Keeping this common lets run_baseline.py stay agent-agnostic: it only needs
to know about TraceBundle/Activation/ActivationGroundTruth, not about
flights, code diffs, or database queries.
"""

import dataclasses
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ActivationGroundTruth:
    """The "answer key" for one activation: what the model's response SHOULD
    say, so run_baseline.py can automatically grade correctness instead of a
    human reading every response.

    "frozen=True" means these objects can't be edited after creation -- once
    the ground truth for an activation is decided, nothing downstream can
    accidentally change it.
    """

    activation_index: int  # which activation this belongs to (0, 1, 2, ...)

    # True only for the ONE activation where we deliberately planted an
    # anomaly (e.g. a flight price crash). Every other activation should be
    # "nothing wrong here" -- so a correct model stays quiet on all of them
    # and speaks up on exactly this one.
    expect_flag: bool

    # Which tracked thing the anomaly happened to -- e.g. "LHR-JFK" for a
    # flight route. None on activations where nothing is wrong.
    anomaly_entity: str | None

    # Which data point (bar/check/commit/etc, depending on the agent) the
    # anomaly was injected at. Mostly useful for debugging the generator,
    # not used directly by the scorer.
    anomaly_bar_index: int | None

    # How big the injected anomaly was (e.g. -0.30 for a 30% price drop).
    anomaly_magnitude: float | None

    detail: str | None = None  # free-text note, not used for scoring


@dataclass(frozen=True)
class Activation:
    """One single "check-in" of the periodic agent -- i.e. one activation of
    the agent, sent as one request to the model."""

    index: int  # 0-based position in the trace (activation 0 is the first ever run)

    # The prompt as one flat string. The model never actually sees this --
    # it's just here so a human (or --dry-run / --smoke-test) can read what
    # was sent without having to reconstruct it from `messages`.
    prompt: str

    # The actual thing sent to the model: a list of {"role": ..., "content": ...}
    # dicts, e.g. [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}].
    # This is the OpenAI/vLLM chat format, fed straight into llm.chat().
    messages: list

    # The answer key for this specific activation (see ActivationGroundTruth above).
    ground_truth: ActivationGroundTruth


@dataclass(frozen=True)
class TraceBundle:
    """The full test: an ordered sequence of activations for one agent, plus
    everything needed to describe/reproduce how it was generated."""

    agent_name: str  # "travel_planner" | "code_reviewer" | "data_analyst"

    # Generic name for "the set of tracked things" -- flight routes for
    # Travel Planner, file names for Code Reviewer, table names for Data
    # Analyst. Kept generic here so this file doesn't need to know which
    # agent it's dealing with.
    tickers_or_entities: list

    num_activations: int  # how many times the agent "checks in" in this trace
    window_size: int      # how much history each activation's prompt shows
    stride: int            # how far the window moves forward each activation
                            # (stride < window_size means consecutive activations
                            # overlap -- old entries age out, new ones appear,
                            # same as a real sliding window of recent data)

    seed: int  # random seed -- same seed always produces the exact same trace,
               # which matters for comparing the --apc on and --apc off runs fairly

    # Where/what/how-big the one deliberately-planted anomaly is. Duplicated
    # here at the TraceBundle level (as well as inside each Activation's
    # ground_truth) purely for convenience -- so you can see at a glance what
    # this trace is testing without digging into every activation.
    anomaly_activation_index: int
    anomaly_entity: str
    anomaly_magnitude: float

    # Every argument that was passed to generate_trace() when this bundle was
    # built -- saved so the trace file is self-describing and reproducible
    # (you can look at a saved trace and know exactly how to regenerate it).
    generation_params: dict

    activations: list  # the actual list of Activation objects, in order


def save_trace(bundle: TraceBundle, path: str) -> None:
    """Writes a trace to a JSON file. Used so the --apc on and --apc off runs
    can be pointed at the exact same generated trace (see run_baseline.py),
    instead of trusting that calling generate_trace() twice produces
    identical results."""
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(bundle), f, indent=2)


def load_trace(path: str) -> TraceBundle:
    """Reads a trace back from JSON. json.load() gives back plain dicts, not
    our dataclasses, so this manually rebuilds the Activation and
    ActivationGroundTruth objects from those dicts before wrapping
    everything back into a TraceBundle."""
    with open(path) as f:
        data = json.load(f)
    activations = [
        Activation(
            index=a["index"],
            prompt=a["prompt"],
            messages=a["messages"],
            ground_truth=ActivationGroundTruth(**a["ground_truth"]),
        )
        for a in data["activations"]
    ]
    data = dict(data)
    data["activations"] = activations
    return TraceBundle(**data)
