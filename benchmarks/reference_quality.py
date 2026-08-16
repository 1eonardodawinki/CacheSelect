"""Deterministic reference-answer metrics for natural-language workloads."""

from __future__ import annotations

import re
import string
from collections import Counter


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


# Match MTRAG's token-recall normalization without external model downloads.
def normalized_answer_tokens(text: str) -> tuple[str, ...]:
    normalized = text.casefold().translate(_PUNCTUATION)
    normalized = _ARTICLES.sub(" ", normalized)
    return tuple(normalized.split())


# Measure how much of the reference answer's word content was recovered.
def reference_token_recall(prediction: str, reference: str) -> float:
    reference_tokens = normalized_answer_tokens(reference)
    if not reference_tokens:
        raise ValueError("reference answer must contain normalized tokens")
    prediction_tokens = normalized_answer_tokens(prediction)
    overlap = Counter(prediction_tokens) & Counter(reference_tokens)
    return sum(overlap.values()) / len(reference_tokens)


# Compute the longest ordered word sequence shared by two answers.
def _longest_common_subsequence_length(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


# Compute a dependency-free word-level ROUGE-L F1 score.
def reference_rouge_l_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalized_answer_tokens(prediction)
    reference_tokens = normalized_answer_tokens(reference)
    if not reference_tokens:
        raise ValueError("reference answer must contain normalized tokens")
    if not prediction_tokens:
        return 0.0
    common = _longest_common_subsequence_length(
        prediction_tokens,
        reference_tokens,
    )
    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if common else 0.0


# Return the lightweight reference metrics recorded for every natural answer.
def score_reference_answer(prediction: str | None, reference: str) -> dict[str, float]:
    text = prediction or ""
    return {
        "token_recall": reference_token_recall(text, reference),
        "rouge_l_f1": reference_rouge_l_f1(text, reference),
    }
