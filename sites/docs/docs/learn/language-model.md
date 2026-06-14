---
title: Language Model
description: The starting point for QuarkLM's closed-world learning.
---

# Language Model

QuarkLM begins from a constrained world:

- a human-authored glossary
- a small grammar
- structured story facts
- an admitted-memory log
- generated lessons derived from those files

The model does not begin with internet-scale knowledge. It does not inherit a
pretrained tokenizer. It does not use external embeddings. The training corpus
is generated from files named in `corpus/ledger.json`.

## Tokenizer

QuarkLM already has its own tokenizer. `closed_world_lm.tokenizer.CharTokenizer`
learns a character vocabulary from admitted corpus text and rejects characters
outside that vocabulary. The current transformer uses this tokenizer.

Future tokenizer work can improve compression, for example with a corpus-derived
subword tokenizer, but it must be trained from admitted text only. A pretrained
vocabulary would cross the same boundary as pretrained weights.

## Transformer

v0.24 adds `closed_world_lm.transformer_char_model`, a tiny decoder-only
transformer built without PyTorch, JAX, Hugging Face, pretrained checkpoints, or
pretrained tokenizers. It starts from random weights and trains with a small
standard-library scalar autodiff engine.

The transformer is not yet the reliable answering path. It is the first
architecture checkpoint for growing beyond the earlier character MLP while
keeping the same closed-world boundary.

## Why So Small?

A small world makes cause and effect inspectable. When the model learns a new
fact, the project can show:

- where the fact was admitted
- how the curriculum changed
- which weights were updated
- which probes checked the behavior
- whether older behavior was preserved

That is the model boundary: growth is gradual, bounded, and auditable.

## What Counts As Knowing?

For this project, a fact is known only when it is inside the admitted corpus and
the promoted responder or learned model can answer its probes. If a fact is not
in the corpus, QuarkLM should answer `unknown`.

This is not a claim about consciousness or subjective selfhood. QuarkLM's
"self" knowledge is operational: dataset boundary, update process, unknown
policy, and current improvement loop.
