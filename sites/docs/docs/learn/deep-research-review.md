---
title: Deep Research Review
description: The v0.70 cross-referenced research and implementation-gap review for QuarkLM.
---

# Deep Research Review

Last reviewed: 2026-06-14.

The full review lives in the repository root at `DEEP_RESEARCH_REVIEW.md`.

v0.70 is a research checkpoint. The earlier forward plan identified the right
direction, but the project needed a deeper cross-reference pass before adding
more mechanics. This page summarizes the decision.

## Conclusion

QuarkLM should not keep adding direct-answer objectives until the learning loop
has the operating system around it:

- experiment registry;
- training recipes;
- corpus hygiene;
- candidate quarantine;
- deterministic closed-world verifier;
- extracted replay planner;
- constraint-first promotion;
- transformer module boundaries.

That is the path that keeps "I learned something new" from becoming
self-contamination. The phrase should mean proposed, quarantined, verified,
admitted, trained, evaluated, and promoted.

## Sources Cross-Referenced

The v0.70 review covers:

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

All sources are design references only. QuarkLM still forbids pretrained
weights, pretrained tokenizers, external embeddings, external datasets, copied
code, and external-model-shaped training data.

## Current Codebase Gap

The strongest local finding is structural. QuarkLM has serious transformer
mechanics now, but `src/closed_world_lm/transformer_char_model.py` owns too
many responsibilities in one 9,494-line module: model, optimizer,
direct-answer objectives, replay planning, snapshot scoring, CLI parsing,
checkpoint writing, and run reporting.

The self-improvement path is cleaner. It already records prompt leakage,
forgetting, exact eval, promotion gates, corpus snapshots, corpus diffs,
attempt archives, and deterministic self-diagnosis. The transformer path needs
the same discipline before another major repair screen.

## Revised Sequence

1. **v0.71:** experiment registry and run-intent schemas. Implemented.
2. **v0.72:** standalone replay planner. Implemented.
3. **v0.73:** corpus hygiene and training-plan artifacts. Implemented.
4. **v0.74:** research implementation map. Implemented.
5. **v0.75:** candidate quarantine. Implemented.
6. **v0.76:** deterministic closed-world verifier. Implemented.
7. **v0.77:** recipes and constraint-first promotion gates. Implemented.
8. **v0.78+:** transformer responsibility refactor, new anti-collapse objective,
   tokenizer growth, or learned
   verifier experiments.

## Operating Rule

No larger transformer screen should run without an experiment-intent artifact,
corpus plan, replay plan, verifier checks, and explicit promotion constraints.
Loss, rank, top-k, and NLL are useful metrics, but they are not promotion
criteria unless retention, leakage, unknown-policy, coverage, diversity, and
contamination gates pass first.

v0.71 satisfies the experiment-intent part of that rule for self-improvement
answer cycles and transformer answer-training screens. v0.72 satisfies the
replay-extraction part by moving replay planning to
`src/closed_world_lm/replay_plan.py`. v0.73 satisfies the corpus-hygiene and
training-plan part by writing `corpus_hygiene.json` and `training_plan.json`.
v0.74 adds the [Research implementation map](./research-implementation-map.md)
so the next mechanics are tied to sources, gaps, and acceptance evidence. v0.75
adds candidate quarantine artifacts and lifecycle state. v0.76 adds
deterministic closed-world verifier checks. v0.77 adds recipes and
constraint-first promotion. The remaining required mechanic before another
larger transformer screen is transformer responsibility refactoring.
