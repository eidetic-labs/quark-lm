"""Constants for the closed-world answer decoder."""

from __future__ import annotations

from answer_model import DEFAULT_RUN_DIR


EOS = "<eos>"
BOS = "<bos>"
DEFAULT_DECODER_RUN_DIR = DEFAULT_RUN_DIR.parent / "answer-decoder-latest"
DECODER_SELF_LEARNING_REPEATS = 55
DECODER_GLOSSARY_REPEATS = 24
