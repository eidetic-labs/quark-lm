---
title: Deep Research Review
description: The v0.70 cross-referenced research and implementation-gap review for QuarkLM.
---

# Deep Research Review

Last reviewed: 2026-06-14.

The full review lives in the repository root at `DEEP_RESEARCH_REVIEW.md`.

v0.70 is a research checkpoint, not a model improvement. The earlier forward
plan named the right direction, but the project needed a deeper cross-reference
pass — primary papers, official open-source mechanics, and the current codebase
read together — before adding more training mechanics. This page records the
decision that pass produced.

## The decision

QuarkLM should stop adding direct-answer objectives until the learning loop has
an operating system around it. A new mechanic is justified only when it sits
inside that system, not when it moves a loss number on its own.

The required pieces are:

| Piece | What it provides |
| --- | --- |
| Experiment registry | A recorded hypothesis, acceptance gate, and decision for every screen. |
| Training recipes | An explicit model, data, objective, optimizer, and replay specification. |
| Corpus hygiene | Duplicate and train/eval overlap checks over the admitted corpus. |
| Candidate quarantine | Generated material held outside training until verified and admitted. |
| Deterministic closed-world verifier | A pass/fail check on the data boundary before evidence is trusted. |
| Extracted replay planner | Replay planning moved out of the model module into its own surface. |
| Constraint-first promotion | Gates that run before any loss, rank, or quality number can count. |
| Transformer module boundaries | The model split into inspectable surfaces rather than one file. |

That is the path that keeps "I learned something new" from meaning
"I generated something new." Inside this system the phrase means proposed,
quarantined, verified, admitted, trained, evaluated, and promoted — in that
order. Generated material is not training data until it is verified against
admitted sources and admitted to `corpus/ledger.json`.

## Sources cross-referenced

The review reads the design literature and official open-source mechanics
together. The clusters are:

- continual-learning surveys for staged updates and catastrophic forgetting;
- replay systems such as Deep Generative Replay, Avalanche, and Reverb;
- self-generated-data methods such as Self-Instruct, STaR, Self-Refine, and
  Reflexion;
- self-feedback and self-judgment risks, including self-bias and reward
  hacking;
- verifiable-reward systems such as Tulu 3 and DeepSeek-R1;
- model-collapse studies on recursive synthetic training;
- small-model and data-centric work such as TinyStories, BabyLM, and SmolLM2;
- transparent open-model practice from Pythia, LLM360, OLMo, OLMo 2, and Dolma;
- implementation references from nanoGPT, minGPT, GPT-NeoX, LLM Foundry,
  LitGPT, Avalanche, Open-Instruct, and Hugging Face tokenizers.

All of these are design references only. None of them is a source of weights or
data. QuarkLM still forbids pretrained weights, pretrained tokenizers, external
embeddings, external datasets, copied code, and external-model-shaped training
data. The version-by-version map from each source cluster to the mechanic it
motivated is kept in the
[Research implementation map](./research-implementation-map.md); the gap matrix
that compares QuarkLM against those mechanics is in the
[Open-source mechanics audit](./open-source-mechanics-audit.md).

## The codebase gap

The strongest local finding is structural rather than about model quality.

QuarkLM has serious transformer mechanics, but at the time of the review
`src/transformer_char_model.py` owned too many responsibilities in one
9,494-line module: the model, the optimizer, the direct-answer objectives,
replay planning, snapshot scoring, CLI parsing, checkpoint writing, and run
reporting. A module that wide cannot be audited screen by screen.

The self-improvement path was already cleaner. It records prompt leakage,
forgetting, exact eval, promotion gates, corpus snapshots, corpus diffs,
attempt archives, and deterministic self-diagnosis. The conclusion is that the
transformer path needs the same discipline — separable surfaces and recorded
evidence — before another large repair screen runs.

## What followed

The review set a sequence of research-control checkpoints, not direct-answer
knobs. Each one builds part of the operating system above, and each is recorded
where its evidence belongs rather than restated here:

| Checkpoint | What it added |
| --- | --- |
| v0.71 | Experiment registry and run-intent schemas. |
| v0.72 | Standalone replay planner in `src/replay_plan.py`. |
| v0.73 | Corpus hygiene and training-plan artifacts (`corpus_hygiene.json`, `training_plan.json`). |
| v0.74 | The [Research implementation map](./research-implementation-map.md), tying mechanics to sources, gaps, and acceptance evidence. |
| v0.75 | Candidate quarantine artifacts and lifecycle state. |
| v0.76 | Deterministic closed-world verifier checks. |
| v0.77 | Recipes and constraint-first promotion gates. |
| v0.78–v0.80 | Transformer experiment, artifact, model/config, checkpoint, and eval surfaces split out of the model module. |

The per-screen direct-answer history that followed — every objective name, its
attempt and acceptance counts, and its rejection evidence — is the job of the
[Transformer screen history](../build/transformer-screen-history.md). The
forward strategy through the current screen is in the
[Forward research plan](./forward-research-plan.md). This page does not
duplicate either; it records why that history is structured as a chain of
rejected diagnostics rather than a sequence of promotions.

## Operating rule

The review leaves one durable rule for every later screen:

No larger transformer screen runs without an experiment-intent artifact, a
corpus plan, a replay plan, verifier checks, and explicit promotion
constraints. Loss, rank, top-k, and NLL are useful metrics, but they are not
promotion criteria. A snapshot is promoted only after retention, leakage,
unknown-policy, coverage, diversity, and contamination gates pass first.

This is why the screens after v0.70 read as a long list of accepted guarded
updates that still reject promotion. Many of them improve coverage or
diagnostics; none has cleared the gate.

## Routing-repair handoff

The open blocker is `branch_diversity_target`. Multi-target eval profiles
collapse to too few predicted branch tokens: the weights learn to emit one
dominant token instead of routing each prompt to its own answer. From v0.112
the failure is classified as a critical `target_routing_gap`, and the screens
through v0.115.0 instrument it rather than clear it — branch routing audits,
logit-prior and centroid-separation summaries, and a hidden-projection margin
candidate. The current candidate and the external research behind it are
detailed in [Branch diversity research](./branch-diversity-research.md).

Retrieval memory answers `219/219` eval probes exactly, with provenance and no
weight updates. That is evidence for the memory-served rail, not for neural
promotion. The transformer remains unpromoted and blocked on branch diversity:
the system can serve every admitted answer while the weights have not yet
learned to route them. `memory-served` is not `weight-consolidated`.
