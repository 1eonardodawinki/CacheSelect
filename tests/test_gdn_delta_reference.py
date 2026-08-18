import math
import unittest

import torch

from benchmarks.gdn_delta_reference import (
    apply_gdn_block_delta_operator,
    build_gdn_block_delta_operator,
    estimate_gdn_delta_cache,
    propagate_gdn_state_delta,
)


# Execute the ordinary GDN recurrence for comparison with delta propagation.
def _run_full_recurrence(
    initial_state: torch.Tensor,
    keys: torch.Tensor,
    queries: torch.Tensor,
    values: torch.Tensor,
    log_decays: torch.Tensor,
    betas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    value_heads = initial_state.shape[0]
    repeats = value_heads // keys.shape[1]
    expanded_keys = keys.repeat_interleave(repeats, dim=1)
    expanded_queries = queries.repeat_interleave(repeats, dim=1)
    scale = 1.0 / math.sqrt(keys.shape[-1])
    state = initial_state
    states = []
    outputs = []
    for token_index in range(keys.shape[0]):
        key = expanded_keys[token_index]
        decayed_state = state * log_decays[token_index].exp()[:, None, None]
        prediction = torch.einsum("hvk,hk->hv", decayed_state, key)
        residual = values[token_index] - prediction
        state = decayed_state + (
            betas[token_index, :, None, None] * residual[:, :, None] * key[:, None, :]
        )
        states.append(state)
        outputs.append(
            torch.einsum("hvk,hk->hv", state, expanded_queries[token_index]) * scale
        )
    return torch.stack(states), torch.stack(outputs)


class GDNDeltaReferenceTests(unittest.TestCase):
    # Quantify the extra cached tensors separately from existing GDN state.
    def test_estimates_block_operator_memory(self) -> None:
        estimate = estimate_gdn_delta_cache(
            block_size=7,
            value_heads=4,
            key_width=3,
            value_width=5,
            gdn_layers=2,
            resident_blocks=10,
            element_bytes=2,
        )

        self.assertEqual(estimate["checkpoint_elements_per_block_layer"], 60)
        self.assertEqual(estimate["transition_elements_per_block_layer"], 36)
        self.assertEqual(estimate["response_elements_per_block_layer"], 84)
        self.assertEqual(estimate["state_bias_elements_per_block_layer"], 60)
        self.assertEqual(estimate["output_bias_elements_per_block_layer"], 140)
        self.assertEqual(estimate["existing_bytes_per_block_layer"], 120)
        self.assertEqual(estimate["auxiliary_bytes_per_block_layer"], 640)
        self.assertEqual(estimate["model_auxiliary_bytes"], 12800)
        self.assertAlmostEqual(
            estimate["auxiliary_to_checkpoint_ratio"],
            16 / 3,
        )

    # Prove delta propagation equals two full runs when later coefficients match.
    def test_matches_difference_between_full_recurrences(self) -> None:
        generator = torch.Generator().manual_seed(7)
        token_count, key_heads, value_heads = 6, 2, 4
        key_width, value_width = 3, 5
        old_state = torch.randn(
            value_heads,
            value_width,
            key_width,
            generator=generator,
            dtype=torch.float64,
        )
        edit_delta = torch.randn(
            value_heads,
            value_width,
            key_width,
            generator=generator,
            dtype=torch.float64,
        )
        keys = torch.randn(
            token_count, key_heads, key_width, generator=generator, dtype=torch.float64
        )
        queries = torch.randn(
            token_count, key_heads, key_width, generator=generator, dtype=torch.float64
        )
        values = torch.randn(
            token_count,
            value_heads,
            value_width,
            generator=generator,
            dtype=torch.float64,
        )
        log_decays = -torch.rand(
            token_count, value_heads, generator=generator, dtype=torch.float64
        )
        betas = torch.rand(
            token_count, value_heads, generator=generator, dtype=torch.float64
        )

        old_states, old_outputs = _run_full_recurrence(
            old_state, keys, queries, values, log_decays, betas
        )
        new_states, new_outputs = _run_full_recurrence(
            old_state + edit_delta, keys, queries, values, log_decays, betas
        )
        delta_states, delta_outputs = propagate_gdn_state_delta(
            edit_delta, keys, queries, log_decays, betas
        )

        torch.testing.assert_close(delta_states, new_states - old_states)
        torch.testing.assert_close(delta_outputs, new_outputs - old_outputs)

    # Keep an unchanged starting state exactly unchanged through propagation.
    def test_zero_edit_has_zero_effect(self) -> None:
        initial = torch.zeros(2, 3, 4, dtype=torch.float64)
        keys = torch.ones(3, 1, 4, dtype=torch.float64)
        states, outputs = propagate_gdn_state_delta(
            initial,
            keys,
            keys,
            -torch.ones(3, 2, dtype=torch.float64),
            torch.full((3, 2), 0.5, dtype=torch.float64),
        )

        self.assertEqual(tuple(states.shape), (3, 2, 3, 4))
        self.assertEqual(tuple(outputs.shape), (3, 2, 3))
        self.assertEqual(torch.count_nonzero(states).item(), 0)
        self.assertEqual(torch.count_nonzero(outputs).item(), 0)

    # Prove the reusable block form equals the direct token-by-token recurrence.
    def test_block_operator_matches_token_delta_propagation(self) -> None:
        generator = torch.Generator().manual_seed(19)
        initial = torch.randn(4, 5, 3, generator=generator, dtype=torch.float64)
        keys = torch.randn(7, 2, 3, generator=generator, dtype=torch.float64)
        queries = torch.randn(7, 2, 3, generator=generator, dtype=torch.float64)
        log_decays = -torch.rand(7, 4, generator=generator, dtype=torch.float64)
        betas = torch.rand(7, 4, generator=generator, dtype=torch.float64)

        direct_states, direct_outputs = propagate_gdn_state_delta(
            initial, keys, queries, log_decays, betas
        )
        operator = build_gdn_block_delta_operator(keys, queries, log_decays, betas)
        final_state, block_outputs = apply_gdn_block_delta_operator(initial, operator)

        torch.testing.assert_close(final_state, direct_states[-1])
        torch.testing.assert_close(block_outputs, direct_outputs)


if __name__ == "__main__":
    unittest.main()
