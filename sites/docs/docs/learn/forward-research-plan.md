---
title: Forward Research Plan
description: The research-backed implementation sequence for QuarkLM's next self-improvement phase.
---

# Forward Research Plan

Last reviewed: 2026-06-14.

The full plan lives in the repository root at `FORWARD_RESEARCH_PLAN.md`.

v0.68 taught a useful but uncomfortable lesson: QuarkLM can improve target-rank
evidence while damaging profile coverage and branch diversity. The next step is
therefore not another direct-answer knob. The next step is the operating system
around training: experiment intent, corpus governance, candidate quarantine,
closed-world verification, replay planning, recipes, and constraint-first
promotion gates.

## What We Reviewed

The v0.69 review cross-references three bodies of evidence:

- continual-learning and replay research;
- self-generated data, self-feedback, and model-collapse research;
- public open-source mechanics from OLMo, Pythia, GPT-NeoX, nanoGPT, minGPT,
  LitGPT, LLM Foundry, Avalanche, Dolma, Open-Instruct, Self-Instruct,
  Self-Refine, and Hugging Face tokenizers.

Those sources are design references only. They do not change QuarkLM's purity
boundary: no pretrained weights, no pretrained tokenizer, no external
embeddings, no copied code, and no unledgered training data.

v0.70 adds the deeper [Deep research review](./deep-research-review.md). It
cross-checks primary papers, official open-source mechanics, and the current
QuarkLM codebase before the next implementation step.

## Main Finding

Mature language-model projects do not improve by secretly changing one training
knob at a time. They make data mixtures, recipes, replay buffers, evaluation
sets, contamination checks, checkpoints, logs, and release artifacts explicit.

For QuarkLM, that means:

- generated lessons must be candidates before they are training data;
- replay must be planned before training, not reconstructed inside a loss;
- every run needs a hypothesis and acceptance gate;
- verifier checks must precede learned self-judgment;
- promotion must reject loss or rank gains that erase coverage, diversity,
  retention, or unknown-policy behavior.

## Implementation Sequence

1. **Experiment registry:** record hypothesis, allowed data, planned artifacts,
   gates, failure criteria, and decision before every run.
2. **Replay extraction:** move profile-aware replay planning out of the
   transformer monolith and preserve the v0.67 behavior with focused tests.
3. **Corpus hygiene:** report source mixtures, duplicate pressure,
   train/eval overlap, generated-candidate ratios, and rare-profile coverage.
4. **Candidate quarantine:** store generated lessons, probes, and repair notes
   as candidates that cannot train weights until admitted.
5. **Closed-world verifier:** start deterministic, then later train a verifier
   only from admitted candidate history and run outcomes.
6. **Recipe layer:** make model, tokenizer, curriculum, replay plan, objective,
   optimizer, snapshot cadence, and promotion gates named and reproducible.
7. **Constraint-first promotion:** compare loss, rank, and top-k only after
   retention, leakage, unknown-policy, target coverage, and diversity pass.

## Near-Term Decision

v0.69 is strategy evidence, v0.70 is deep research evidence, and v0.71-v0.83
are the first operating-system implementation steps. None of those are
model-quality promotion evidence. v0.81 returns to objective-repair work under
the narrower operating surfaces with profile target-share anti-collapse
pressure; v0.82 screens that pressure and rejects it on branch diversity.
v0.83 adds prompt-specific ownership margins and rejects the screen because
trained snapshots still lose target-token coverage. v0.84 adds baseline replay
anchors and rejects the screen because trained snapshots preserve only half of
the baseline QA/heldout coverage floor. v0.85 adds baseline-floor update gating
and rejects the screen because the guard preserves the floor only by rejecting
all attempted direct-answer updates. v0.86 adds adaptive baseline-floor retries
and rejects the screen because all `200/200` retry attempts still violate the
floor, so v0.87 should change the update shape rather than only the step size.

v0.71 implements experiment registry and run-intent schemas. v0.72 extracts
replay planning into `src/closed_world_lm/replay_plan.py` while preserving the
profile-aware replay behavior. v0.73 adds corpus hygiene and training-plan
artifacts for source mixture, duplicates, train/eval overlap, candidate ratio,
rare-profile coverage, allowed data sources, planned artifacts, and replay-plan
summaries. v0.74 adds the
[Research implementation map](./research-implementation-map.md), which ties
each next mechanic to source clusters, public implementation patterns, QuarkLM
gaps, and acceptance evidence before more code is added. v0.75 implements
candidate quarantine artifacts and lifecycle states. v0.76 implements
deterministic closed-world verifier checks. v0.77 implements recipes and
constraint-first promotion. v0.78 implements transformer experiment/artifact
surfaces, trainer utilities, and a direct-answer objective catalog. v0.79
implements transformer model/config and checkpoint metadata surfaces. v0.80
implements transformer eval/checkpoint-load surfaces. v0.81 implements
`branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood`
as the first post-surface anti-collapse objective. v0.82 screens it at
`runs/transformer-answer-v0.82-fullstack-profile-target-share-smoke-dim4-context80/`
and rejects it because trained snapshots still collapse QA and heldout branch
diversity before rank gains can be trusted. v0.83 adds
`branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood`
and screens it at
`runs/transformer-answer-v0.83-fullstack-prompt-ownership-smoke-dim4-context80/`.
The focused mechanic works, but the full screen remains rejected because rank
gains still require target-token coverage collapse. v0.84 adds
`branch-balanced-context-profile-baseline-anchored-prompt-ownership-target-share-preserving-deficit-unlikelihood`
and screens it at
`runs/transformer-answer-v0.84-fullstack-baseline-anchored-prompt-ownership-smoke-dim4-context80/`.
The run records `562` active baseline prediction anchors and avoids the v0.83
`0.0` coverage collapse, but QA/heldout target-token coverage only reaches
`0.125` against the `0.25` baseline floor.

v0.85 adds
`branch-balanced-context-profile-baseline-floor-gated-prompt-ownership-target-share-preserving-deficit-unlikelihood`
and screens it at
`runs/transformer-answer-v0.85-fullstack-baseline-floor-gated-prompt-ownership-smoke-dim4-context80/`.
The run records `562` active baseline prediction anchors and checks `50/50`
attempted updates under a baseline-floor guard. The guard rejects all `50`
attempts, preserving QA/heldout coverage at `0.25` but accepting no weight
updates.

v0.86 adds
`branch-balanced-context-profile-baseline-floor-adaptive-prompt-ownership-target-share-preserving-deficit-unlikelihood`
and screens it at
`runs/transformer-answer-v0.86-fullstack-baseline-floor-adaptive-prompt-ownership-smoke-dim4-context80/`.
The run tries learning-rate scales `1.0`, `0.25`, `0.05`, and `0.01` for each
guarded direct-answer step. It records `200` attempted retry updates, rejects
all `200`, preserves QA/heldout coverage at `0.25`, and accepts no weight
updates.
