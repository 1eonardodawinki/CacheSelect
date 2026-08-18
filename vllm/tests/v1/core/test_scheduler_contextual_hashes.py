# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler


# Build the minimum scheduler state needed by the pure hash-routing helper.
def _scheduler(*, has_mamba_layers: bool, mode: str) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.has_mamba_layers = has_mamba_layers
    scheduler.cache_config = SimpleNamespace(
        mamba_cache_mode=mode,
        gdn_delta_cache_capacity=2,
        mamba_block_size=4,
    )
    scheduler.hash_block_size = 2
    # Real hybrid models can pad the common scheduler page far beyond GDN's
    # checkpoint interval; the helper must ignore this physical page geometry.
    scheduler.block_size = 8
    return scheduler


# Verify logical GDN checkpoints do not inherit the padded scheduler page size.
def test_resolves_contextual_hashes_at_gdn_block_size() -> None:
    scheduler = _scheduler(has_mamba_layers=True, mode="all")
    request = SimpleNamespace(block_hashes=[b"h0", b"h1", b"h2", b"h3"])

    assert scheduler._get_contextual_block_hashes(request) == (b"h1", b"h3")


# Verify ordinary and single-state requests do not pay the metadata cost.
def test_omits_contextual_hashes_outside_all_state_mamba() -> None:
    request = SimpleNamespace(block_hashes=[b"h0", b"h1"])

    assert _scheduler(
        has_mamba_layers=False, mode="all"
    )._get_contextual_block_hashes(request) == ()
    assert _scheduler(
        has_mamba_layers=True, mode="align"
    )._get_contextual_block_hashes(request) == ()
