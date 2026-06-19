"""Tokenizer selection and vocab expansion for transformer training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tokenizer import CharTokenizer
from tokenizer_artifacts import (
    propose_closed_world_subword_tokenizer,
    write_tokenizer_artifacts,
)
from tokenizer_artifact_validation import validate_tokenizer_artifacts
from tokenizer_protocol import TokenizerProtocol
from transformer_model import transformer_config_from_args
from transformer_vocab_expansion_audit import audit_vocab_expansion_parity
from transformer_vocab_expansion import expand_weights_for_tokenizer


def training_tokenizer(
    args: argparse.Namespace,
    train_text: str,
    model_cls: Any,
) -> TokenizerProtocol:
    args._training_text_for_vocab_audit = train_text
    if args.resume_checkpoint is None:
        return _fresh_training_tokenizer(args, train_text)
    _model, checkpoint_tokenizer = model_cls.load(args.resume_checkpoint)
    if checkpoint_tokenizer is None:
        raise ValueError("resume checkpoint does not contain a tokenizer")
    if args.tokenizer == "closed-world-subword":
        return _subword_training_tokenizer(args, train_text, checkpoint_tokenizer)
    args.tokenizer_manifest_hash = None
    return checkpoint_tokenizer.extend(train_text)


def initialize_transformer_for_training_command(
    args: argparse.Namespace,
    tokenizer: TokenizerProtocol,
    model_cls: Any,
) -> tuple[Any, dict[str, Any]]:
    if args.resume_checkpoint is None:
        config = transformer_config_from_args(args, tokenizer.vocab_size)
        return model_cls.init_random(config), {"resumed": False}
    model, checkpoint_tokenizer = model_cls.load(args.resume_checkpoint)
    if checkpoint_tokenizer is None:
        raise ValueError("resume checkpoint does not contain a tokenizer")
    checkpoint_config = transformer_config_from_args(args, checkpoint_tokenizer.vocab_size)
    if asdict(model.config) != asdict(checkpoint_config):
        raise ValueError("resume checkpoint config does not match requested transformer config")
    if not tokenizer.extends(checkpoint_tokenizer):
        raise ValueError(
            "resume checkpoint tokenizer is not an append-only prefix of "
            "the admitted training tokenizer"
        )
    vocab_expansion_audit = None
    if tokenizer.vocab_size > checkpoint_tokenizer.vocab_size:
        base_model_payload = model.to_dict(checkpoint_tokenizer)
        base_model = model_cls(checkpoint_config, base_model_payload["weights"])
        if _can_resize_model_vocab(checkpoint_tokenizer, tokenizer):
            model.resize_vocab(tokenizer.vocab_size)
        else:
            expanded_weights = expand_weights_for_tokenizer(
                base_model_payload["weights"],
                checkpoint_tokenizer,
                tokenizer,
            )
            expanded_config = transformer_config_from_args(args, tokenizer.vocab_size)
            model = model_cls(expanded_config, expanded_weights)
        vocab_expansion_audit = audit_vocab_expansion_parity(
            base_model=base_model,
            expanded_model=model,
            base_tokenizer=checkpoint_tokenizer,
            expanded_tokenizer=tokenizer,
            train_text=getattr(args, "_training_text_for_vocab_audit", ""),
            context_size=args.context_size,
        )
        if not vocab_expansion_audit["passed"]:
            raise ValueError("vocabulary expansion changed old-token logits")
    return model, {
        "resumed": True,
        "resume_checkpoint": str(args.resume_checkpoint),
        "tokenizer_extended": tokenizer.vocab_size > checkpoint_tokenizer.vocab_size,
        "previous_vocab_size": checkpoint_tokenizer.vocab_size,
        "vocab_size": tokenizer.vocab_size,
        "added_tokens": tokenizer.tokens[checkpoint_tokenizer.vocab_size :],
        "vocab_expansion_audit": vocab_expansion_audit,
    }


def _fresh_training_tokenizer(
    args: argparse.Namespace,
    train_text: str,
) -> TokenizerProtocol:
    if args.tokenizer == "char":
        args.tokenizer_manifest_hash = None
        return CharTokenizer.train(train_text)
    return _subword_training_tokenizer(args, train_text, None)


def _can_resize_model_vocab(base_tokenizer: Any, expanded_tokenizer: Any) -> bool:
    return (
        base_tokenizer.tokenizer_type == "char"
        and expanded_tokenizer.tokenizer_type == "char"
    )


def _subword_training_tokenizer(
    args: argparse.Namespace,
    train_text: str,
    base_tokenizer: Any | None,
) -> TokenizerProtocol:
    proposal = propose_closed_world_subword_tokenizer(
        train_text,
        source_files=[str(_training_source_path(args))],
        max_token_chars=args.tokenizer_max_token_chars,
        max_new_tokens=args.tokenizer_max_new_tokens,
        base_tokenizer=base_tokenizer,
    )
    validate_tokenizer_artifacts(
        proposal["manifest"],
        proposal["report"],
        manifest_hash=proposal["manifest_hash"],
    )
    manifest_path = _artifact_path(
        args.tokenizer_manifest,
        args.run / "tokenizer_manifest.json",
    )
    report_path = _artifact_path(
        args.tokenizer_report,
        args.run / "tokenizer_report.json",
    )
    write_tokenizer_artifacts(
        manifest_path,
        report_path,
        proposal["manifest"],
        proposal["report"],
    )
    args.tokenizer_manifest_hash = proposal["manifest_hash"]
    return proposal["tokenizer"]


def _artifact_path(value: str | None, default: Path) -> Path:
    return Path(value) if value else default


def _training_source_path(args: argparse.Namespace) -> Path:
    source = getattr(args, "corpus", None) or getattr(args, "train_text", None)
    if source is None:
        return Path("unknown")
    return Path(source)
