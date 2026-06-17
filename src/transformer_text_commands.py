"""Classic transformer text training and evaluation commands."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from curriculum import DEFAULT_OUTPUT_DIR, build_curriculum, write_curriculum
from neural_char_model import context_before
from tokenizer import CharTokenizer
from transformer_eval import (
    build_transformer_eval_report,
    eval_candidates_from_records,
    load_probe_records,
    score_transformer_evals,
    write_eval_report,
    write_eval_samples,
)
from transformer_experiment import (
    transformer_training_recipe as build_transformer_training_recipe,
)
from transformer_math import average_nll
from transformer_model import (
    TRANSFORMER_ARCHITECTURE,
    TRANSFORMER_CHECKPOINT_FORMAT,
    TRANSFORMER_TOKENIZER,
    generation_config_from_args,
    optimization_config_from_args,
    transformer_config_from_args,
    transformer_run_metadata,
)
from transformer_optimizer import load_optimizer_state, save_optimizer_state
from transformer_paths import DEFAULT_PROBES


def ensure_curriculum(corpus_path: Path, valid_path: Path) -> None:
    if corpus_path.exists() and valid_path.exists():
        return
    curriculum = build_curriculum()
    write_curriculum(curriculum, DEFAULT_OUTPUT_DIR)


def transformer_training_recipe(
    args: argparse.Namespace,
    tokenizer: CharTokenizer,
    planned_artifacts: list[Path],
    acceptance_gates: list[dict[str, Any]],
    replay_plan_path: Path | None = None,
) -> dict[str, Any]:
    return build_transformer_training_recipe(
        args,
        tokenizer,
        planned_artifacts,
        acceptance_gates,
        asdict(transformer_config_from_args(args, tokenizer.vocab_size)),
        asdict(optimization_config_from_args(args)),
        asdict(generation_config_from_args(args)),
        replay_plan_path,
    )


def initialize_transformer_for_training_command(
    args: argparse.Namespace,
    tokenizer: CharTokenizer,
    model_cls: Any,
) -> tuple[Any, dict[str, Any]]:
    if args.resume_checkpoint is None:
        config = transformer_config_from_args(args, tokenizer.vocab_size)
        return model_cls.init_random(config), {"resumed": False}
    model, checkpoint_tokenizer = model_cls.load(args.resume_checkpoint)
    if checkpoint_tokenizer is None:
        raise ValueError("resume checkpoint does not contain a tokenizer")
    if checkpoint_tokenizer.to_dict() != tokenizer.to_dict():
        raise ValueError("resume checkpoint tokenizer does not match admitted training tokenizer")
    requested_config = transformer_config_from_args(args, tokenizer.vocab_size)
    if asdict(model.config) != asdict(requested_config):
        raise ValueError("resume checkpoint config does not match requested transformer config")
    return model, {
        "resumed": True,
        "resume_checkpoint": str(args.resume_checkpoint),
    }


def train_transformer_command(args: argparse.Namespace, model_cls: Any) -> dict[str, Any]:
    ensure_curriculum(args.corpus, args.valid)
    train_text = args.corpus.read_text(encoding="utf-8")
    valid_text = args.valid.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.train(train_text)
    train_ids = tokenizer.encode(train_text)
    valid_ids = tokenizer.encode(valid_text)
    model, resume_metadata = initialize_transformer_for_training_command(args, tokenizer, model_cls)
    optimizer = load_optimizer_state(
        args.resume_optimizer,
        optimization_config_from_args(args),
    )
    model.active_optimizer = optimizer
    rng = random.Random(args.seed)
    args.run.mkdir(parents=True, exist_ok=True)
    history_path = args.run / "transformer_metrics.jsonl"

    def write_history(step: int, train_nll: float | None) -> dict[str, Any]:
        record = {
            "step": step,
            "train_nll": train_nll,
            "valid_nll": average_nll(model, valid_ids, tokenizer.pad_id, args.valid_limit),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    baseline = write_history(step=0, train_nll=None)
    running_loss = 0.0
    last_history = baseline
    last_history_step = 0
    for step in range(1, args.steps + 1):
        position = rng.randrange(len(train_ids))
        context = context_before(train_ids, position, args.context_size, tokenizer.pad_id)
        running_loss += model.train_step(context, train_ids[position], args.learning_rate)
        if args.eval_every > 0 and step % args.eval_every == 0:
            train_loss = running_loss / args.eval_every
            last_history = write_history(step=step, train_nll=train_loss)
            last_history_step = step
            print(
                f"step={step} train_nll={train_loss:.4f} "
                f"valid_nll={last_history['valid_nll']:.4f}"
            )
            running_loss = 0.0

    if last_history_step != args.steps:
        last_history = write_history(step=args.steps, train_nll=None)

    checkpoint_path = args.run / "transformer.json"
    optimizer_path = args.run / "optimizer_state.json"
    save_optimizer_state(optimizer_path, optimizer)
    checkpoint_metadata = transformer_run_metadata(
        args,
        tokenizer,
        optimizer,
        "language-model",
        resume_metadata,
    )
    model.save(checkpoint_path, tokenizer, checkpoint_metadata)
    tokenizer.save(args.run / "tokenizer.json")
    metrics = {
        "architecture": TRANSFORMER_ARCHITECTURE,
        "checkpoint": str(checkpoint_path),
        "checkpoint_format": TRANSFORMER_CHECKPOINT_FORMAT,
        "optimizer_state": str(optimizer_path),
        "optimizer": optimizer.summary(),
        "resume": resume_metadata,
        "history": str(history_path),
        "steps": args.steps,
        "train_chars": len(train_text),
        "valid_chars": len(valid_text),
        "vocab_size": tokenizer.vocab_size,
        "context_size": args.context_size,
        "embedding_dim": args.embedding_dim,
        "feedforward_dim": args.feedforward_dim,
        "num_layers": args.num_layers,
        "attention_heads": args.attention_heads,
        "use_layer_norm": args.use_layer_norm,
        "use_pre_layer_norm": args.use_pre_layer_norm,
        "use_rms_norm": args.use_rms_norm,
        "layer_norm_epsilon": args.layer_norm_epsilon,
        "use_gated_mlp": args.use_gated_mlp,
        "tie_output_embeddings": args.tie_output_embeddings,
        "use_rotary_positions": args.use_rotary_positions,
        "use_kv_cache_path": args.use_kv_cache_path,
        "use_context_mean": args.use_context_mean,
        "use_context_projection": args.use_context_projection,
        "use_prompt_prefix_projection": args.use_prompt_prefix_projection,
        "use_prompt_position_projection": args.use_prompt_position_projection,
        "prompt_position_projection_scale": args.prompt_position_projection_scale,
        "use_prompt_attention_summary": args.use_prompt_attention_summary,
        "baseline_valid_nll": baseline["valid_nll"],
        "final_valid_nll": last_history["valid_nll"],
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "tokenizer": TRANSFORMER_TOKENIZER,
    }
    with (args.run / "transformer_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"saved {checkpoint_path}")
    return metrics


def eval_transformer_command(args: argparse.Namespace, model_cls: Any) -> dict[str, Any]:
    model, tokenizer = model_cls.load(args.checkpoint)
    if tokenizer is None:
        raise ValueError("checkpoint does not contain a tokenizer")
    generation_config = generation_config_from_args(args)
    probe_paths = args.probe if args.probe is not None else DEFAULT_PROBES
    probe_records = load_probe_records(probe_paths)
    candidates = eval_candidates_from_records(probe_records)
    scored_by_eval = score_transformer_evals(
        model,
        tokenizer,
        probe_records,
        args.max_new_chars,
        generation_config,
        candidates,
    )
    result = build_transformer_eval_report(
        args.checkpoint,
        probe_paths,
        probe_records,
        scored_by_eval,
        candidates,
        generation_config,
        args.samples_jsonl,
    )
    if args.samples_jsonl:
        write_eval_samples(args.samples_jsonl, scored_by_eval)
    if args.json:
        write_eval_report(args.json, result)
    return result
