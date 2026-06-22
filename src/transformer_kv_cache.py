"""Regime-aware append-valid KV cache (Phase 3) -- scalar reference.

Pure Python (NO torch). This is the audited reference for the decode-path KV
cache; the torch runtime mirrors it. It is deliberately narrow so that a later
phase cannot inherit a wrong assumption: it caches K/V for LAYER 0 ONLY.

WHY LAYER 0 ONLY (the central, adversarially-confirmed correctness fact)
-----------------------------------------------------------------------
Under the thesis decode geometry (``use_absolute_rope`` + dropped learned
pos-embed addend + ``use_rotary_positions``):

  * The layer-0 input for a window slot is the bare token embedding (the learned
    position-embedding addend is DROPPED). RoPE rotates K by the token's ABSOLUTE
    stream position, and V is never rotated. So layer-0 K[p] / V[p] depend ONLY
    on (token at p, p) -- they are WRITE-ONCE: a token at a fixed absolute
    position p produces the same layer-0 K/V on every step, even as the
    right-anchored window slides and p moves to a lower slot. Caching is valid.

  * For UPPER layers it is FALSE. ``_forward_full_block_floats`` does slot-relative
    causal attention over ``range(context_size)``, and ``make_context_positioned``
    advances ``start`` as the stream grows, so a token at a fixed absolute
    position p slides to a LOWER slot and its causal prefix shrinks from the left.
    Its upper-layer K/V INPUT (the residual stream feeding layer 1+) therefore
    changes step-to-step. A write-once upper-layer cache would serve a stale row
    and silently flip a token. THEREFORE upper layers are RECOMPUTED every step.

The saving grows with ``context_size`` (layer 0's historical K/V re-projection is
skipped), NOT with ``num_layers``. There is NO depth-compounding saving under this
scope. For ``num_layers == 1`` (the default and the validated thesis config) layer
0 IS the whole model, so the cache covers the full model.

The windowed position-projection readout reads no K/V -- it re-windows the residual
rows by slot -- and is therefore UNCHANGED and UNAFFECTED by this cache.

ACCUMULATION ORDER
------------------
``_causal_attention`` (transformer_lm_attention.py) sums the weighted values over
``range(position + 1)`` -- i.e. ascending by slot, which under a right-anchored
window is ascending by absolute position. ``assemble`` reproduces exactly that
order: it walks window slots 0..last_position and maps each to its absolute
position, so the float32 weighted-sum order matches the recompute bit-for-bit. A
dict/insertion-order reorder would silently break the f32 band.
"""

from __future__ import annotations

from typing import Callable


# Sentinel: see neural_char_ops.POSITION_PAD_SENTINEL. Duplicated as a local
# constant (this module is torch-free and import-light) -- a left-pad slot carries
# a negative absolute position and must NEVER be cached (its key would collide
# across steps, since every short-prompt step reuses the same -1 sentinel).
_PAD_POSITION_FLOOR = 0


class Layer0KVCache:
    """Append-valid write-once K/V cache for LAYER 0 ONLY (scalar reference).

    Two position-keyed stores. ``store(position, key_row, value_row)`` writes a row
    keyed by its ABSOLUTE stream position, but ONLY for a non-negative position (a
    pad slot is silently skipped). ``assemble`` returns the cached K and V rows for a
    requested ordered list of absolute positions, in the SAME order they are passed
    -- the caller passes window slots ascending so the weighted-sum order matches the
    recompute exactly.

    By construction there is NO layer index and NO upper-layer path: the cache simply
    cannot hold anything but layer 0, so a future caller cannot accidentally cache an
    upper layer through it.
    """

    LAYER_INDEX = 0  # structural: this cache is layer-0-only, full stop.

    def __init__(self) -> None:
        self._keys: dict[int, list[float]] = {}
        self._values: dict[int, list[float]] = {}
        self.hits = 0
        self.writes = 0
        self.pad_skips = 0

    def reset(self) -> None:
        self._keys.clear()
        self._values.clear()
        self.hits = 0
        self.writes = 0
        self.pad_skips = 0

    def has(self, position: int) -> bool:
        return position in self._keys

    def store(self, position: int, key_row: list[float], value_row: list[float]) -> None:
        """Write-once store, gated on a non-negative absolute position.

        A negative position is a left-pad sentinel: skip it (never cache a pad). The
        write is idempotent for a real position -- under the write-once invariant the
        same (token, position) always yields the same K/V, so a re-store of an existing
        key is a no-op overwrite with identical content.
        """

        if position < _PAD_POSITION_FLOOR:
            self.pad_skips += 1
            return
        self._keys[position] = list(key_row)
        self._values[position] = list(value_row)
        self.writes += 1

    def store_new_token(
        self, position: int, key_row: list[float], value_row: list[float]
    ) -> None:
        """Store the freshly-computed K/V for the newest token (the last window slot)."""

        self.store(position, key_row, value_row)

    def has_all_history(self, positions: list[int]) -> bool:
        """Whether every NON-PAD position in ``positions`` is already cached."""

        return all(p < _PAD_POSITION_FLOOR or p in self._keys for p in positions)

    def assemble(
        self,
        positions: list[int],
        compute_row: Callable[[int], tuple[list[float], list[float]]],
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Return (keys, values) for ``positions`` IN THE GIVEN ORDER.

        ``positions`` is the absolute position of each window slot, slot 0..last
        ASCENDING -- so the returned rows are in the exact order ``_causal_attention``
        sums them. For each slot: a cached non-pad position is served from the store (a
        hit); any other slot (a pad, or an uncached real position) is filled by
        ``compute_row(slot_index)``, which the caller supplies to recompute that one
        row from the live forward. Pad slots are never cached, so they are always
        recomputed -- but their RoPE rotation is the identity and they carry the same
        token, so the recompute is exact.
        """

        keys: list[list[float]] = []
        values: list[list[float]] = []
        for slot_index, position in enumerate(positions):
            if position >= _PAD_POSITION_FLOOR and position in self._keys:
                keys.append(list(self._keys[position]))
                values.append(list(self._values[position]))
                self.hits += 1
            else:
                key_row, value_row = compute_row(slot_index)
                keys.append(list(key_row))
                values.append(list(value_row))
        return keys, values
