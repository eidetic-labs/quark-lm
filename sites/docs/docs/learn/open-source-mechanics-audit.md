---
title: Open-Source Mechanics Audit
description: What QuarkLM learns from open-source LLM and continual-learning mechanics without copying code or data.
---

# Open-Source Mechanics Audit

Last reviewed: 2026-06-14.

QuarkLM uses open-source projects and papers as design references, not as
sources of model weights, tokenizers, data, embeddings, or copied
implementations. The full audit lives in the repository root at
`MECHANICS_AUDIT.md`.

## What Changed

The earlier `STRUCTURE_AUDIT.md` looked at transformer shape. This mechanics
audit looks at the surrounding system:

- trainer boundaries;
- replay plans;
- profile-aware continual learning;
- checkpoint selection;
- tokenizer-growth artifacts;
- self-generated candidate filtering;
- transparency and evidence release discipline.

## Main Finding

QuarkLM's next bottleneck is not another global branch-loss term. The next
useful change is trainer mechanics: explicit profile-aware replay plans,
profile-local coverage deficits, profile-local preservation, and checkpoint
selection that treats coverage, unknown-policy, leakage, and retention as
constraints before ranking snapshots by loss or target rank.

## Reference Map

| Source | What QuarkLM studies | What QuarkLM does not take |
| --- | --- | --- |
| nanoGPT and minGPT | compact trainer/model boundaries, checkpoint cadence, optimizer state, generation traces | code, weights, GPT-2 imports, datasets, tokenizer state |
| LitGPT | config-driven decoder-only recipes, norm/rotary/KV-cache mechanics | implementation code, recipes as training data, model weights |
| Hugging Face tokenizers | tokenizer pipeline concepts, special-token and alignment artifacts | pretrained vocabularies or merge tables |
| Avalanche | continual-learning streams, replay strategies, evaluation plugins | library dependency, benchmark data, external pretrained models |
| Self-Instruct, STaR, Reflexion | candidate generation, filtering, and memory-before-weight-update separation | external-model generated training material |
| LLM360, OLMo, OLMo 2 | transparent code/data/checkpoint/log/recipe practice and data-mixture reporting | open training corpora, weights, or external checkpoints |

## Required Direction

Before the next full-stack direct-answer repair run, the branch replay path
should:

1. carry profile keys through branch records;
2. compute missing targets per profile instead of globally;
3. preserve represented coverage per profile;
4. emit a replay-plan artifact with profile counts, target sets, represented
   targets, deficits, and coverage floors;
5. test that one profile's improvement cannot mask another profile's deficit.

This keeps self-improvement aligned with the closed-world claim: new behavior
must be trained from admitted data, measured by profile, and rejected when it
improves one metric by erasing another.
