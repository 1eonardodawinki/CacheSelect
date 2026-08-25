# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math

import pytest

from vllm.v1.worker.gpu.mlp_repair_model import MLPRepairModel


def test_loads_and_runs_exported_mlp(tmp_path) -> None:
    payload = {
        "schema_version": 1,
        "feature_names": ["first", "second"],
        "selected_repair_threshold": 0.7,
        "standardizer": {"mean": [1.0, 2.0], "scale": [2.0, 4.0]},
        "layers": [
            {
                "weights": [[1.0, -1.0], [2.0, 1.0]],
                "bias": [0.5, 0.0],
                "activation": "relu",
            },
            {
                "weights": [[1.0], [2.0]],
                "bias": [-1.0],
                "activation": "logistic",
            },
        ],
    }
    path = tmp_path / "mlp.json"
    path.write_text(json.dumps(payload))

    model = MLPRepairModel.from_json(path)
    probability = model.predict_repair_probability({"first": 3.0, "second": 6.0})

    assert probability == pytest.approx(1 / (1 + math.exp(-2.5)))
    assert model.should_repair({"first": 3.0, "second": 6.0})


def test_rejects_mismatched_standardizer() -> None:
    with pytest.raises(ValueError, match="sizes must match"):
        MLPRepairModel(
            {
                "schema_version": 1,
                "feature_names": ["only"],
                "selected_repair_threshold": 0.5,
                "standardizer": {"mean": [0.0], "scale": [1.0, 2.0]},
                "layers": [],
            }
        )
