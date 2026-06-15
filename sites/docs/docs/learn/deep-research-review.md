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
8. **v0.78:** transformer experiment/artifact surfaces, trainer utilities,
   and objective catalog. Implemented.
9. **v0.79:** transformer model/config and checkpoint metadata surfaces.
   Implemented.
10. **v0.80:** transformer eval/checkpoint-load surfaces. Implemented.
11. **v0.81:** profile target-share anti-collapse objective. Implemented.
12. **v0.82:** profile target-share full-stack screen. Implemented and
    rejected.
13. **v0.83:** prompt-specific branch ownership. Implemented and rejected.
14. **v0.84:** baseline replay anchors. Implemented and rejected.
15. **v0.85:** baseline-floor update gating. Implemented and rejected.
16. **v0.86:** adaptive baseline-floor retries. Implemented and rejected.
17. **v0.87:** baseline-floor repair retries. Implemented and rejected.
18. **v0.88+:** a floor-preserving objective under the full baseline
    target-token floor, tokenizer growth, or learned verifier experiments.

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
constraint-first promotion. v0.78 adds transformer experiment/artifact
surfaces, trainer utilities, and an objective catalog. v0.79 adds transformer
model/config and checkpoint metadata surfaces. v0.80 adds transformer
eval/checkpoint-load surfaces. v0.81 adds balanced profile target-share
pressure to the preserving-deficit direct-answer objective. v0.82 screens that
objective, preserves coverage only by restoring step `0`, and rejects trained
snapshots that collapse QA and heldout branch diversity. v0.83 adds
prompt-specific sibling-target ownership margins, but the full screen still
restores step `0` because trained snapshots lose target-token coverage. v0.84
adds baseline replay anchors; trained snapshots avoid the v0.83 zero-coverage
collapse but still restore step `0` because coverage reaches only `0.125`
against the `0.25` floor. v0.85 adds baseline-floor update gating; it preserves
the floor by rejecting `50/50` unsafe attempted updates, so the next repair must
produce accepted updates under the guard. v0.86 adds adaptive baseline-floor
retries across learning-rate scales `1.0`, `0.25`, `0.05`, and `0.01`; it
rejects all `200/200` attempted retry updates, showing step size alone is not
the missing mechanic and the next repair must change update shape under the
same baseline floor. v0.87 adds one bounded baseline-covered anchor repair
after each failed retry; it records `227` repair anchors and rejects all
`200/200` repaired attempts, showing post-update repair is also not the missing
mechanic. The next repair must make the objective floor-preserving before
optimizer application.
