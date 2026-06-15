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

v0.67 implements the first direct-answer replay path with these mechanics:

1. Profile keys are carried through branch records.
2. Missing targets are computed per profile instead of globally.
3. Represented coverage is preserved per profile.
4. A replay-plan artifact records profile counts, target sets, represented
   targets, deficits, and coverage floors.
5. Focused tests verify that one profile's improvement cannot mask another
   profile's deficit.

The bounded smoke run
`runs/transformer-answer-v0.67-profile-aware-replay-plan-smoke-dim4-context80/`
wrote `direct_answer_replay_plan.json` for `9144` branch records across `21`
profiles and passed the branch-context gate. It is mechanics-readiness evidence
only: it made the next full-stack repair run measurable against profile-aware
constraints instead of another global replay target set.

v0.68 runs that full-stack screen. It improves QA and heldout target rank during
training, but the gains come with target-token coverage and predicted diversity
regressions, so best-snapshot scoring restores step `0`. The next mechanics
move is anti-collapse preservation inside profile-local replay constraints.

v0.70 adds the [Deep research review](./deep-research-review.md), which expands
this audit into a fuller literature, implementation, and QuarkLM-codebase gap
review. It keeps the same engineering direction and moves the implementation
sequence to experiment registry first. v0.71 implements that registry and
v0.72 extracts replay planning into a standalone module. Corpus hygiene is the
next mechanics step. v0.73 implements corpus hygiene and training-plan
artifacts. v0.74 adds the research implementation map, v0.75 implements
candidate quarantine artifacts with source-backed acceptance criteria, and
v0.76 implements deterministic closed-world verifier checks. v0.77 implements
recipe artifacts and constraint-first promotion reports. v0.78 implements the
first transformer responsibility split for experiment/artifact surfaces,
trainer utilities, and the direct-answer objective catalog. v0.79 implements
model/config and checkpoint metadata surfaces. v0.80 implements
eval/checkpoint-load surfaces. v0.81 implements profile target-share pressure
inside the profile-aware preserving-deficit direct-answer objective. v0.82
screens that objective and rejects it on branch diversity under the
constraint-first gate. v0.83 adds prompt-specific ownership margins, but the
full screen remains rejected because trained snapshots still collapse
target-token coverage. v0.84 adds baseline replay anchors, improves trained
coverage relative to v0.83, and remains rejected because it still misses the
baseline coverage floor. v0.85 adds baseline-floor update gating, preserves the
floor by rejecting all attempted unsafe updates, and remains rejected because no
update is accepted. v0.86 adds adaptive baseline-floor retries, rejects
`200/200` retry attempts across smaller learning-rate scales, and shows the next
mechanics change must alter update shape while preserving the same floor. v0.87
adds baseline-covered repair retries, rejects `200/200` repaired attempts, and
shows the next mechanics change must put the floor-preserving constraint inside
the objective before optimizer application. v0.88 does that with objective-side
floor anchors, rejects `200/200` objective-shaped attempts, and shows the next
mechanics change should prove floor-stabilization updates before branch pressure
is added back.

This keeps self-improvement aligned with the closed-world claim: new behavior
must be trained from admitted data, measured by profile, and rejected when it
improves one metric by erasing another.
