# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler


# Build the minimum scheduler state needed by the pure hash-routing helper.
def _scheduler(*, has_mamba_layers: bool, mode: str) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.has_mamba_layers = has_mamba_layers
    scheduler.cache_config = SimpleNamespace(mamba_cache_mode=mode)
    scheduler.hash_block_size = 2
    scheduler.block_size = 4
    return scheduler


# Verify coarser logical blocks use the last chained fine-grained block hash.
def test_resolves_contextual_hashes_at_scheduler_block_size() -> None:
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
