---
title: Secure
description: Keep QuarkLM inside its closed-world boundary.
slug: /secure/
---

# Secure

Security for QuarkLM is not about access control or network hardening. It is
about keeping an epistemic boundary intact. The project is only interesting if
it can say where each piece of knowledge came from, refuse to learn from
anything unaccounted-for, and admit plainly when it does not know. The pages
under Secure define that boundary and the checks that hold it in place.

Three claims have to stay true for the rest of the documentation to mean
anything:

- the neural weights learn only from the admitted, ledgered corpus;
- held-out evaluation prompts never leak into training;
- a fact outside the corpus is answered `unknown`, not guessed.

Each claim is one page below.

## What the boundary protects

QuarkLM separates *answering* from *learning*, and only the learning path can
change neural weights — see [Build](../build/index.md). The boundary exists so
that what enters the weights can be traced to a ledgered source, and so that the
evidence for any claim stays auditable. Retrieval answering a probe is
`memory-served`: it proves the admitted corpus contains the answer. It does not
prove the transformer `weight-consolidated` that answer. The secure surfaces
keep those two states from blurring into one.

| Surface | Protects against | Enforced by |
| --- | --- | --- |
| Purity boundary | Pretrained weights, tokenizers, embeddings, or unledgered text entering training. | `corpus/ledger.json` as the admission gate. |
| Prompt leakage | Held-out evaluation prompts being copied into lessons the model trains on. | The self-improvement report's leakage audit. |
| Unknown policy | Answers invented from nearby surface forms or the surrounding world. | The deterministic responder's `unknown` rail. |

## Purity boundary

Allowed training data must come from the admitted corpus or from corpus-derived
lessons generated from ledgered sources. QuarkLM imports no pretrained weights,
no pretrained tokenizer, no external embeddings, and no unledgered training text.
A pretrained vocabulary would cross the same line as pretrained weights, so even
the character tokenizer learns from admitted text only.

`corpus/ledger.json` is the gate. It names each source and whether that source
is allowed for curriculum generation or training. Generated material — a lesson,
probe, repair, or memory proposal — is not training data until it is verified
against admitted sources and admitted to the ledger. See
[Purity boundary](./purity-boundary.md).

## Prompt leakage

The corpus is allowed to teach a held-out fact; it is not allowed to teach the
exact form of a held-out evaluation prompt. If the evaluation prompt forms were
copied into lesson files, a passing eval would no longer distinguish learned
behavior from memorized test items.

QuarkLM keeps the two apart by auditing for leakage in the self-improvement
report. The audit covers held-out facts and held-out ownership prompts, and the
expected result is zero leaked protected prompts. See
[Prompt leakage](./prompt-leakage.md).

## Unknown policy

When a fact is outside the admitted corpus, QuarkLM answers `unknown.` rather
than producing a plausible-looking guess. For training-data-status questions,
known admitted facts answer `yes` and facts outside the corpus answer `no`.

This rail keeps the prototype honest about its own boundary: it must not invent
answers from nearby surface forms or from the surrounding world. An honest
`unknown` is the correct output, not a failure. See
[Unknown policy](./unknown-policy.md).

## Read

| Page | Covers |
| --- | --- |
| [Purity boundary](./purity-boundary.md) | What QuarkLM may and may not train on, and why the ledger is the gate. |
| [Prompt leakage](./prompt-leakage.md) | How held-out evaluation prompts are kept out of training lessons. |
| [Unknown policy](./unknown-policy.md) | How QuarkLM answers — and declines to answer — outside the corpus. |
