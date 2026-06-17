---
title: Purity Boundary
description: What QuarkLM may and may not train on.
---

# Purity Boundary

The purity boundary is the line between knowledge QuarkLM's neural weights are
allowed to learn from and everything else. Its neural weights learn only from the
admitted, ledgered corpus. Nothing arrives pretrained, and no text becomes
training input by being typed, retrieved, or generated. This page states the
prohibitions, the ledger that enforces them, and the one distinction that is easy
to miss: studying open-source structure is allowed; importing open-source weights,
tokenizers, embeddings, or text is not.

## What may not enter

Four things are excluded from every part of the model that touches neural
weights: training, validation, tokenizer fitting, and repair data.

| Excluded input | Why it is excluded |
| --- | --- |
| Pretrained weights | They carry knowledge from outside the corpus that cannot be traced to an admitted source. |
| Pretrained tokenizers | A pretrained vocabulary encodes a world the corpus never admitted; it crosses the same boundary as pretrained weights. |
| External embeddings | They inject learned representations the corpus did not produce. |
| Unledgered training text | Any text not named and allowed in `corpus/ledger.json` is not training data, regardless of how true it is. |

These four are not a code-comment policy. They are the same four boundary flags
the deterministic [closed-world verifier](../operate/closed-world-verifier.md)
checks before a training plan is trusted: `pretrained_weights`,
`pretrained_tokenizer`, `external_embeddings`, and `unledgered_training_text`
must all be `false`.

## The ledger is the gate

`corpus/ledger.json` is the single source that decides what is admitted and what
each admitted source may be used for. It carries a `purity_boundary` block that
records the four prohibitions above and a `training_data_policy` stating that only
curriculum generated from allowed source files in the ledger may be used for
training. It then lists every source with two independent permissions:

| Field | Controls |
| --- | --- |
| `allowed_for_curriculum_generation` | Whether `curriculum` may derive training and validation text from the source. |
| `allowed_for_training` | Whether the source's derived material may reach a weight update. |

Both permissions default to closed. A source is allowed only when its ledger
entry says so explicitly. The human-authored glossary, the grammar and story
facts, and the admitted-memory log are marked allowed for both; everything else
in the ledger is not.

## Admitted is not the same as trainable

A source can be named in the ledger and still be excluded from training. The
evaluation probe sets — the known, unknown, held-out, paraphrase, ownership,
self, learning, admission, and glossary probes — are all ledgered sources, but
each carries `allowed_for_curriculum_generation: false` and
`allowed_for_training: false`. They are admitted so their provenance is recorded,
not so the model can learn from them. Letting an eval probe into training would
teach the model the test, which is the leak the
[prompt leakage](./prompt-leakage.md) discipline guards against.

```text
corpus/ledger.json
  glossary-v0     curriculum: yes   training: yes
  grammar-v0      curriculum: yes   training: yes
  admissions-v0   curriculum: yes   training: yes
  *-probes-v0     curriculum: no    training: no   (admitted for provenance only)
```

## Generated material is not yet trainable

The boundary also holds for material QuarkLM produces itself. A curriculum
lesson, a candidate, or a repair proposal derived from admitted sources is not
training data until it has been verified against admitted sources and admitted to
the ledger. Until then it sits in
[candidate quarantine](../operate/candidate-quarantine.md), where the verifier
confirms a ledger admission link before the material can become training-eligible.
Generation does not bypass admission; it feeds it.

This is also why retrieval memory answering a probe is not evidence that the
weights learned it. Retrieval serves admitted knowledge with provenance and moves
no weights — it is `memory-served`, not `weight-consolidated`. The two states are
kept separate everywhere; see [Language model](../learn/language-model.md).

## Structure may be studied; substance may not be imported

The boundary restricts imported content, not knowledge of how transformers are
built. Per `STRUCTURE_AUDIT.md`, QuarkLM may study open-source model, trainer,
tokenizer, and checkpoint *structure* — config layout, attention and residual
shapes, tokenizer pipeline stages, evaluation artifact formats — and adopt those
shapes when each is recorded in docs, tests, and run evidence. It must not import
external weights, tokenizers, embeddings, datasets, or training text.

The character tokenizer (`tokenizer.CharTokenizer`) is the working example: its
pipeline structure can follow open-source references, but its vocabulary is
trained from admitted corpus text and rejects out-of-vocabulary characters. Any
future subword tokenizer must be a separate artifact trained from admitted text
only — a pretrained vocabulary would cross the boundary that pretrained weights
do.

## Rule

New training data must be admitted to `corpus/ledger.json` or generated from
admitted corpus files and then admitted in turn. A source's two ledger
permissions decide whether it may reach curriculum and weights; both are closed
until the ledger opens them. Evaluation probes may be ledgered for provenance,
but they are not training data unless the ledger explicitly allows it.
