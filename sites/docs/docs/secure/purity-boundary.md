---
title: Purity Boundary
description: What QuarkLM may and may not train on.
---

# Purity Boundary

QuarkLM does not use:

- pretrained weights
- pretrained tokenizers
- external embeddings
- unledgered training text

Allowed training data must come from the admitted corpus or corpus-derived
lessons generated from ledgered sources.

`corpus/ledger.json` is the gate. It names each source and whether the source is
allowed for curriculum generation or training.
