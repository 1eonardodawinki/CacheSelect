# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dependency-free inference for an exported CacheSelect repair MLP."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class MLPRepairModel:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported MLP repair model schema")
        self.feature_names = tuple(payload["feature_names"])
        standardizer = payload["standardizer"]
        self.mean = tuple(map(float, standardizer["mean"]))
        self.scale = tuple(map(float, standardizer["scale"]))
        if not self.feature_names or not (
            len(self.feature_names) == len(self.mean) == len(self.scale)
        ):
            raise ValueError("MLP feature and standardizer sizes must match")
        if any(scale == 0 for scale in self.scale):
            raise ValueError("MLP standardizer scales must be non-zero")
        self.layers = tuple(payload["layers"])
        self.repair_threshold = float(payload["selected_repair_threshold"])

    @classmethod
    def from_json(cls, path: str | Path) -> MLPRepairModel:
        return cls(json.loads(Path(path).read_text()))

    def predict_repair_probability(self, features: Mapping[str, float]) -> float:
        values = [
            (float(features[name]) - mean) / scale
            for name, mean, scale in zip(
                self.feature_names, self.mean, self.scale, strict=True
            )
        ]
        if not all(map(math.isfinite, values)):
            raise ValueError("MLP features must be finite")
        for layer in self.layers:
            weights = layer["weights"]
            bias = layer["bias"]
            if len(weights) != len(values):
                raise ValueError("MLP layer input size does not match")
            values = [
                sum(
                    value * float(row[column])
                    for value, row in zip(values, weights, strict=True)
                )
                + float(offset)
                for column, offset in enumerate(bias)
            ]
            activation = layer["activation"]
            if activation == "relu":
                values = [max(0.0, value) for value in values]
            elif activation == "logistic":
                values = [
                    1.0 / (1.0 + math.exp(-value))
                    if value >= 0
                    else math.exp(value) / (1.0 + math.exp(value))
                    for value in values
                ]
            else:
                raise ValueError(f"unsupported MLP activation: {activation}")
        if len(values) != 1:
            raise ValueError("binary repair MLP must produce one value")
        if not math.isfinite(values[0]):
            raise ValueError("binary repair MLP produced a non-finite value")
        return values[0]

    def should_repair(self, features: Mapping[str, float]) -> bool:
        return self.predict_repair_probability(features) >= self.repair_threshold
