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

## Memory-Native Philosophy

Large language models typically compress broad public and licensed corpora into
weights first, then use retrieval, fine-tuning, adapters, or prompting to steer
that pretrained base toward a task. That path buys fluency and breadth with
enormous compute, but the resulting knowledge boundary is difficult to audit:
it is hard to prove exactly where a fact entered, why a behavior changed, or
whether a later update erased an earlier one.

QuarkLM takes the smaller path on purpose. Knowledge enters as a lesson, becomes
part of the admitted corpus, is served immediately by retrieval memory when the
corpus can answer it, and only becomes a weight-update candidate after it has a
verifiable source. The lifecycle is:

```text
new lesson -> corpus -> retrieval memory -> training candidates -> guarded weight update -> evaluation -> accepted or rejected
```

In this design, retrieval memory is not a shortcut around learning. It is the
first reliable expression of the model's closed world. Weight consolidation is
the slower step that teaches the neural model language behavior, routing,
paraphrase tolerance, compression, and generalization over that world. An update
is accepted only when evaluation shows it improved the target behavior without
violating the corpus boundary or regressing prior evidence.

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

The transformer is not yet the reliable answering path. It is the weight
consolidation path: the component that should gradually learn from admitted
training candidates after retrieval memory has made the knowledge available and
evaluation can reject harmful updates.

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
