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
floor. v0.87 adds baseline-covered repair retries and rejects the screen because
all `200/200` repaired attempts still violate the floor, setting up the v0.88
objective screen. v0.88 adds
objective-side baseline-floor anchors and rejects the screen because all
`200/200` objective-shaped attempts still violate the floor. v0.89 removes
branch-diversity pressure and trains only baseline-covered floor anchors, but
all `200/200` stabilization-only attempts still violate the floor, so v0.90
adds guard diagnostics before branch-diversity pressure is added back. v0.90
shows all `200` rejected attempts are stabilization-shaped, every adaptive scale
fails `50` times, `heldout` violates all attempts, and the worst deficit is
`0.25` on `learning`. v0.91 covers all `227` floor anchors across `12`
profile-target groups and still rejects all `200/200` attempts. v0.92 changes
the repair shape to sequential source-profile floor batches, rejects all
`2000` profile-local attempts, and records `200` no-effective-update outer
attempts, so the next repair must isolate floor-preserving weight movement
rather than only broaden anchor coverage or reorder profiles. v0.93 adds
calibrated scales below `0.01` plus coverage-only guard probes and accepts one
nonzero `bridge:owner` source-profile update at scale `0.0025`, while model
promotion remains blocked on branch diversity. v0.94 adds profile-scale memory,
accepts `8` source-profile updates across `60` profile-scale attempts, and
keeps promotion blocked on branch diversity. v0.95 adds diversity-aware
profile-scale acceptance, accepts `5` score-improving source-profile updates
across `58` profile-scale attempts, rejects `11` floor-preserving score
regressions, and keeps promotion blocked on branch diversity.

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

v0.87 adds
`branch-balanced-context-profile-baseline-floor-repaired-prompt-ownership-target-share-preserving-deficit-unlikelihood`
and screens it at
`runs/transformer-answer-v0.87-fullstack-baseline-floor-repaired-prompt-ownership-clean-smoke-dim4-context80/`.
The run records `227` repair anchors and applies one bounded baseline-covered
anchor repair before each failed adaptive retry is accepted or rejected. It
records `200` repaired attempts, rejects all `200`, preserves QA/heldout
coverage at `0.25`, and accepts no weight updates.

v0.88 adds
`branch-balanced-context-profile-baseline-floor-objective-prompt-ownership-target-share-preserving-deficit-unlikelihood`
and screens it at
`runs/transformer-answer-v0.88-fullstack-baseline-floor-objective-prompt-ownership-smoke-dim4-context80/`.
The run records `227` objective-side floor anchors and includes a balanced
anchor batch in the same loss and backward pass as branch-diversity pressure.
It records `200` objective anchor batches, rejects all `200` attempted updates,
preserves QA/heldout coverage at `0.25`, and accepts no weight updates.

v0.89 adds `branch-context-profile-baseline-floor-stabilization-unlikelihood`
and screens it at
`runs/transformer-answer-v0.89-fullstack-baseline-floor-stabilization-smoke-dim4-context80/`.
The run removes branch-diversity pressure from guarded attempts and trains only
baseline-covered floor anchors. It records `227` stabilization anchors, `200`
stabilization anchor batches, rejects all `200` attempted updates, preserves
QA/heldout coverage at `0.25`, and accepts no weight updates.

v0.90 adds baseline-floor rejection diagnostics and screens them at
`runs/transformer-answer-v0.90-fullstack-baseline-floor-stabilization-diagnostics-smoke-dim4-context80/`.
The run records rejected update-shape counts, rejected learning-rate scale
counts, violation profile counts, compact floor diagnostic samples, and the
worst rejected floor violation. It still rejects `200/200` attempts, but it now
identifies the next repair targets: `heldout`, `admissions`, `glossary`, `qa`,
and the worst-deficit `learning` profile.

v0.91 adds
`branch-context-profile-baseline-floor-profile-targeted-stabilization-unlikelihood`
and screens it at
`runs/transformer-answer-v0.91-fullstack-baseline-floor-profile-targeted-stabilization-smoke-dim4-context80/`.
The run covers `227` floor anchors across `12` profile-target groups on every
guarded attempt, but still rejects `200/200` profile-targeted updates with the
same violation profile counts as v0.90.

v0.92 adds
`branch-context-profile-baseline-floor-sequential-profile-stabilization-unlikelihood`
and screens it at
`runs/transformer-answer-v0.92-fullstack-baseline-floor-sequential-profile-stabilization-smoke-dim4-context80/`.
The run covers `10` source-profile floor groups sequentially on every guarded
attempt, rejects all `2000` profile-local attempts, and records `200`
no-effective-update outer attempts. This shifts the next repair from profile
ordering toward smaller or more isolated floor-preserving weight movement.

v0.93 adds
`branch-context-profile-baseline-floor-calibrated-sequential-profile-stabilization-unlikelihood`
and screens it at
`runs/transformer-answer-v0.93-baseline-floor-calibrated-sequential-profile-stabilization-step1-dim4-context80/`.
The run records calibrated scales down to `0.0001`, coverage-only guard probes,
`50` profile-local attempts, `49` profile-local rejections, and one accepted
nonzero `bridge:owner` update at scale `0.0025`. The next repair should expand
accepted calibrated movement beyond one source profile.

v0.94 adds
`branch-context-profile-baseline-floor-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood`
and screens it at
`runs/transformer-answer-v0.94-baseline-floor-profile-scale-calibrated-sequential-stabilization-step1-dim4-context80/`.
The run searches calibrated scales per source profile, records `60`
profile-scale attempts, accepts `8` source-profile updates, rejects `52`
profile-scale attempts, and preserves the baseline floor. The next repair
should turn this safe movement into branch-diverse behavior.

v0.95 adds
`branch-context-profile-baseline-floor-diversity-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood`
and screens it at
`runs/transformer-answer-v0.95-baseline-floor-diversity-profile-scale-calibrated-sequential-stabilization-configured-step1-dim4-context80/`.
The run keeps the calibrated per-profile scale search, records `58`
profile-scale attempts, accepts `5` score-improving source-profile updates,
rejects `42` floor regressions and `11` floor-preserving score regressions, and
preserves the baseline floor. The next repair should convert non-regressive
profile movement into full branch-diversity target coverage.
