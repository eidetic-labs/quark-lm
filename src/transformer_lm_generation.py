"""Generation and checkpoint I/O methods for TinyTransformerLM."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from neural_char_model import make_context
from tokenizer import CharTokenizer
from transformer_checkpoint import load_checkpoint_payload
from transformer_math import (
    generation_distribution,
    matrix_to_floats,
    sample_from_probs,
    vector_to_floats,
)
from transformer_model import (
    GenerationConfig,
    TransformerConfig,
    checkpoint_header,
    validate_generation_config,
)


class TransformerGenerationIOMixin:
    def generate(
        self,
        tokenizer: CharTokenizer,
        prompt: str,
        max_new_chars: int,
        temperature: float = 0.0,
        stop_at: str | None = None,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> str:
        config = GenerationConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return self.generate_with_trace(
            tokenizer,
            prompt,
            max_new_chars,
            config,
            stop_at=stop_at,
        )["text"]

    def generate_with_trace(
        self,
        tokenizer: CharTokenizer,
        prompt: str,
        max_new_chars: int,
        config: GenerationConfig | None = None,
        stop_at: str | None = None,
    ) -> dict[str, Any]:
        config = config or GenerationConfig()
        validate_generation_config(config)
        ids = tokenizer.encode(prompt)
        generated: list[int] = []
        rng = random.Random(self.config.seed + len(prompt))
        trace: list[dict[str, Any]] = []
        cache_enabled = config.use_kv_cache or self.config.use_kv_cache_path
        cache_events: list[dict[str, Any]] = []
        for _ in range(max_new_chars):
            context = make_context(ids, self.config.context_size, tokenizer.pad_id)
            if cache_enabled:
                cache_events.append(
                    {
                        "context_length": len(context),
                        "source_token_count": len(ids),
                        "sliding_window": len(ids) > self.config.context_size,
                    }
                )
            probs = self.predict(context)
            filtered_probs = generation_distribution(
                probs,
                generated,
                config,
            )
            if config.temperature <= 0:
                next_id = max(
                    range(len(filtered_probs)),
                    key=lambda index: filtered_probs[index],
                )
            else:
                next_id = sample_from_probs(filtered_probs, 1.0, rng)
            top_tokens = sorted(
                range(len(filtered_probs)),
                key=lambda index: filtered_probs[index],
                reverse=True,
            )[: config.trace_top_tokens]
            trace.append(
                {
                    "step": len(generated) + 1,
                    "context": tokenizer.decode(context),
                    "token_id": next_id,
                    "token": tokenizer.itos[next_id],
                    "probability": filtered_probs[next_id],
                    "raw_probability": probs[next_id],
                    "top_tokens": [
                        {
                            "token_id": token_id,
                            "token": tokenizer.itos[token_id],
                            "probability": filtered_probs[token_id],
                            "raw_probability": probs[token_id],
                        }
                        for token_id in top_tokens
                    ],
                }
            )
            ids.append(next_id)
            if stop_at is not None and tokenizer.itos[next_id] == stop_at:
                break
            generated.append(next_id)
        return {
            "text": tokenizer.decode(generated),
            "trace": trace,
            "generation_config": asdict(config),
            "cache": {
                "enabled": cache_enabled,
                "mode": "rolling-context-kv-aware" if cache_enabled else "disabled",
                "events": cache_events,
            },
        }

    def to_dict(
        self,
        tokenizer: CharTokenizer | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **checkpoint_header(self.config),
            "weights": {
                "token_embeddings": matrix_to_floats(self.token_embeddings),
                "position_embeddings": matrix_to_floats(self.position_embeddings),
                "wq": matrix_to_floats(self.wq),
                "bq": vector_to_floats(self.bq),
                "wk": matrix_to_floats(self.wk),
                "bk": vector_to_floats(self.bk),
                "wv": matrix_to_floats(self.wv),
                "bv": vector_to_floats(self.bv),
                "wo": matrix_to_floats(self.wo),
                "bo": vector_to_floats(self.bo),
                "w1": matrix_to_floats(self.w1),
                "b1": vector_to_floats(self.b1),
                "w_gate": matrix_to_floats(self.w_gate),
                "b_gate": vector_to_floats(self.b_gate),
                "w2": matrix_to_floats(self.w2),
                "b2": vector_to_floats(self.b2),
                "wout": matrix_to_floats(self.wout),
                "bout": vector_to_floats(self.bout),
                "context_projection_w": matrix_to_floats(self.context_projection_w),
                "context_projection_b": vector_to_floats(self.context_projection_b),
                "prompt_prefix_projection_w": matrix_to_floats(
                    self.prompt_prefix_projection_w
                ),
                "prompt_prefix_projection_b": vector_to_floats(
                    self.prompt_prefix_projection_b
                ),
                "prompt_position_projection_w": [
                    matrix_to_floats(position_weights)
                    for position_weights in self.prompt_position_projection_w
                ],
                "prompt_position_projection_b": vector_to_floats(
                    self.prompt_position_projection_b
                ),
                "prompt_summary_query": vector_to_floats(self.prompt_summary_query),
                "prompt_summary_w": matrix_to_floats(self.prompt_summary_w),
                "prompt_summary_b": vector_to_floats(self.prompt_summary_b),
                "ln1_gain": vector_to_floats(self.ln1_gain),
                "ln1_bias": vector_to_floats(self.ln1_bias),
                "ln2_gain": vector_to_floats(self.ln2_gain),
                "ln2_bias": vector_to_floats(self.ln2_bias),
                "final_ln_gain": vector_to_floats(self.final_ln_gain),
                "final_ln_bias": vector_to_floats(self.final_ln_bias),
                "extra_layers": [
                    self._block_to_floats(block)
                    for block in self.extra_blocks
                ],
            },
        }
        if metadata is not None:
            payload["metadata"] = metadata
        if tokenizer is not None:
            payload["tokenizer"] = tokenizer.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> tuple["TinyTransformerLM", CharTokenizer | None]:
        config = TransformerConfig(**payload["config"])
        model = cls(config, payload["weights"])
        tokenizer = None
        if "tokenizer" in payload:
            tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
        return model, tokenizer

    def save(
        self,
        path: Path,
        tokenizer: CharTokenizer | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(tokenizer, metadata), handle)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> tuple["TinyTransformerLM", CharTokenizer | None]:
        return cls.from_dict(load_checkpoint_payload(path))
