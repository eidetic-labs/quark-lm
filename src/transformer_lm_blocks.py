"""Attention-block orchestration for TinyTransformerLM."""

from __future__ import annotations

from typing import Any

from autograd import Scalar
from transformer_lm_attention import TransformerAttentionMixin
from transformer_lm_context_summaries import TransformerContextSummaryMixin
from transformer_lm_feedforward import TransformerFeedForwardMixin
from transformer_math import (
    linear_floats,
    linear_scalars,
)


class TransformerBlockMixin(
    TransformerContextSummaryMixin,
    TransformerFeedForwardMixin,
    TransformerAttentionMixin,
):
    def _forward_full_block_scalars(
        self,
        x: list[list[Scalar]],
        block: dict[str, Any],
        positions: list[int] | None = None,
    ) -> list[list[Scalar]]:
        attention_input = self._attention_input_scalars(x, block)
        q = [linear_scalars(row, block["wq"], block["bq"]) for row in attention_input]
        k = [linear_scalars(row, block["wk"], block["bk"]) for row in attention_input]
        v = [linear_scalars(row, block["wv"], block["bv"]) for row in attention_input]
        if self.config.use_rotary_positions:
            pos = positions if self.config.use_absolute_rope else None
            q = self._apply_rotary_scalars(q, pos)
            k = self._apply_rotary_scalars(k, pos)
        outputs = []
        for position in range(self.config.context_size):
            attended = self._causal_attention_scalars(q, k, v, position)
            projected = linear_scalars(attended, block["wo"], block["bo"])
            hidden = [
                x[position][dim] + projected[dim]
                for dim in range(self.config.embedding_dim)
            ]
            outputs.append(self._feed_forward_scalars(hidden, block))
        return outputs

    def _forward_final_block_floats(
        self,
        x: list[list[float]],
        block: dict[str, Any],
        context: list[int],
        positions: list[int] | None = None,
        cache: Any | None = None,
    ) -> list[float]:
        attention_input = self._attention_input_floats(x, block)
        q = [linear_floats(row, block["wq"], block["bq"]) for row in attention_input]
        if self.config.use_rotary_positions:
            pos = positions if self.config.use_absolute_rope else None
            q = self._apply_rotary_floats(q, pos)
        last_position = self.config.context_size - 1
        # LAYER-0-ONLY append-valid KV cache: when present (and this is the single
        # block, i.e. the whole model is layer 0), serve historical rotated-K / raw-V
        # from the cache and recompute only the newest token's row. ``cache=None`` (the
        # default + every non-layer-0 call) is byte-identical to the prior path.
        k, v = self._layer0_kv_floats(
            attention_input, block, positions, last_position, cache
        )
        attended = self._causal_attention_floats(q, k, v, last_position)
        projected = linear_floats(attended, block["wo"], block["bo"])
        hidden = [
            x[last_position][dim] + projected[dim]
            for dim in range(self.config.embedding_dim)
        ]
        hidden = self._add_final_context_summaries_floats(
            hidden,
            x,
            context,
            last_position,
        )
        return self._feed_forward_floats(hidden, block)

    def _forward_full_block_floats(
        self,
        x: list[list[float]],
        block: dict[str, Any],
        positions: list[int] | None = None,
        cache: Any | None = None,
    ) -> list[list[float]]:
        attention_input = self._attention_input_floats(x, block)
        q = [linear_floats(row, block["wq"], block["bq"]) for row in attention_input]
        if self.config.use_rotary_positions:
            pos = positions if self.config.use_absolute_rope else None
            q = self._apply_rotary_floats(q, pos)
        last_position = self.config.context_size - 1
        # LAYER-0-ONLY cache: ``cache`` is non-None ONLY for layer 0 (the caller passes
        # it only on the first block). Upper layers always pass ``cache=None`` and
        # recompute K/V every step, because a token's UPPER-layer K/V input changes as
        # the window slides -- write-once holds for layer 0 only. ``cache=None`` is
        # byte-identical to the prior path. The K/V here cover all context_size slots,
        # but only the newest row is recomputed when cache hits exist; the historical
        # rows are served from the store (saving layer 0's K/V re-projection).
        k, v = self._layer0_kv_floats(
            attention_input, block, positions, last_position, cache
        )
        outputs = []
        for position in range(self.config.context_size):
            attended = self._causal_attention_floats(q, k, v, position)
            projected = linear_floats(attended, block["wo"], block["bo"])
            hidden = [
                x[position][dim] + projected[dim]
                for dim in range(self.config.embedding_dim)
            ]
            outputs.append(self._feed_forward_floats(hidden, block))
        return outputs

    def _layer0_kv_floats(
        self,
        attention_input: list[list[float]],
        block: dict[str, Any],
        positions: list[int] | None,
        last_position: int,
        cache: Any | None,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Compute (rotated-K, raw-V) rows for layer 0, optionally via the cache.

        REGIME GATE (the central correctness fact): layer-0 K/V is write-once ONLY
        under the thesis geometry -- ``use_absolute_rope`` (the learned pos-embed addend
        is DROPPED) AND ``use_rotary_positions`` (RoPE keys by ABSOLUTE position). In any
        other regime the layer-0 INPUT at a fixed absolute position changes as the window
        slides (a slot-keyed pos-embed addend, or slot-keyed RoPE), so serving a cached
        row would be a stale-row bug. Outside the regime we IGNORE the cache and fully
        recompute, so the decode path is byte-identical to cache-off regardless of the
        flag. This is what makes the cache *regime-aware*.

        In-regime, with a cache: compute K/V for the NEWEST token only (the last slot),
        rotate it, store it under its absolute position, then assemble the full K/V from
        the cache in ascending-slot order -- the order ``_causal_attention_floats`` sums
        them, so the weighted-sum order is bit-identical. Any uncached slot (a left-pad
        sentinel, or the first time a real position is seen) is recomputed inline via
        ``compute_row``; pad slots are never written (their key would collide).

        ``cache=None`` (the default, every upper-layer + non-decode call) is byte-for-byte
        the prior inline computation: project every row to K/V then rotate K.
        """

        write_once_regime = (
            self.config.use_rotary_positions and self.config.use_absolute_rope
        )
        if not write_once_regime:
            # Out of regime: the cache is not append-valid here, so never serve from it.
            cache = None
        pos = positions if write_once_regime else None

        def compute_row(slot_index: int) -> tuple[list[float], list[float]]:
            row = attention_input[slot_index]
            key_row = linear_floats(row, block["wk"], block["bk"])
            value_row = linear_floats(row, block["wv"], block["bv"])
            if self.config.use_rotary_positions:
                # Rotate this one row by its OWN slot's angle. The full path rotates
                # slot i by positions[i] (absolute) or i (slot-keyed); pass the explicit
                # angle so a single-row rotation cannot fall back to enumerate index 0.
                angle_pos = pos[slot_index] if pos is not None else slot_index
                key_row = self._apply_rotary_floats([key_row], [angle_pos])[0]
            return key_row, value_row

        if cache is None:
            k = [linear_floats(row, block["wk"], block["bk"]) for row in attention_input]
            v = [linear_floats(row, block["wv"], block["bv"]) for row in attention_input]
            if self.config.use_rotary_positions:
                k = self._apply_rotary_floats(k, pos)
            return k, v

        # Cache the newest token (the last slot). ``positions`` carries its absolute
        # index; a pad position is silently skipped by the store.
        new_key, new_value = compute_row(last_position)
        if positions is not None:
            cache.store_new_token(positions[last_position], new_key, new_value)

        def assemble_row(slot_index: int) -> tuple[list[float], list[float]]:
            if slot_index == last_position:
                return new_key, new_value
            return compute_row(slot_index)

        slot_positions = (
            positions
            if positions is not None
            else list(range(self.config.context_size))
        )
        return cache.assemble(slot_positions, assemble_row)
