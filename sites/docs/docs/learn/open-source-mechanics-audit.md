---
title: Open-Source Mechanics Audit
description: What QuarkLM learns from open-source LLM and continual-learning mechanics without copying code or data.
---

# Open-Source Mechanics Audit

Last reviewed: 2026-06-14.

QuarkLM reads open-source projects and papers as design references for how a
training and continual-learning system should be *shaped*. It does not take
model weights, tokenizers, data, embeddings, or copied implementations from any
of them. The full audit lives in the repository root at `MECHANICS_AUDIT.md`;
this page is the durable summary.

The distinction is the same one drawn in [Purity boundary](../secure/purity-boundary.md):
studying the structure of an open model is allowed, importing anything it
learned is not. A reference can suggest where a trainer boundary should sit. It
cannot become a vocabulary, a checkpoint, or a line of training text.

## Why this audit exists

An earlier `STRUCTURE_AUDIT.md` looked at the transformer itself — its shape,
attention block, and head. This mechanics audit looks at the system *around* the
model, where the next useful work was found to be:

- trainer boundaries;
- replay plans;
- profile-aware continual learning;
- checkpoint selection;
- tokenizer-growth artifacts;
- self-generated candidate filtering;
- transparency and evidence-release discipline.

The main finding is that QuarkLM's next bottleneck is not another global
branch-loss term. The next useful change is trainer mechanics: explicit
profile-aware replay plans, profile-local coverage deficits, profile-local
preservation, and checkpoint selection that treats coverage, unknown-policy,
leakage, and retention as constraints *before* ranking snapshots by loss or
target rank.

## Reference map

Each row records what QuarkLM studies and, just as importantly, what it refuses
to take across the boundary.

| Source | What QuarkLM studies | What QuarkLM does not take |
| --- | --- | --- |
| nanoGPT and minGPT | compact trainer/model boundaries, checkpoint cadence, optimizer state, generation traces | code, weights, GPT-2 imports, datasets, tokenizer state |
| LitGPT | config-driven decoder-only recipes, norm/rotary/KV-cache mechanics | implementation code, recipes as training data, model weights |
| Hugging Face tokenizers | tokenizer pipeline concepts, special-token and alignment artifacts | pretrained vocabularies or merge tables |
| Avalanche | continual-learning streams, replay strategies, evaluation plugins | library dependency, benchmark data, external pretrained models |
| Self-Instruct, STaR, Reflexion | candidate generation, filtering, and memory-before-weight-update separation | external-model generated training material |
| LLM360, OLMo, OLMo 2 | transparent code/data/checkpoint/log/recipe practice and data-mixture reporting | open training corpora, weights, or external checkpoints |

The bottom-right column is the load-bearing one. Self-generated material in
particular is not training data until it is verified against admitted sources
and admitted to the ledger; see [Candidate quarantine](../operate/candidate-quarantine.md).

## What the audit asked for

The audit named a concrete trainer change: replay planning had to become
profile-aware. The mechanics it required were:

1. Profile keys are carried through branch records.
2. Missing targets are computed per profile instead of globally.
3. Represented coverage is preserved per profile.
4. A replay-plan artifact records profile counts, target sets, represented
   targets, deficits, and coverage floors.
5. Focused tests verify that one profile's improvement cannot mask another
   profile's deficit.

The first implementation wrote `direct_answer_replay_plan.json` over `9144`
branch records across `21` profiles and passed the branch-context gate. It moved
no weights and promoted no snapshot. It was mechanics-readiness evidence only:
it made the next full-stack repair run measurable against profile-aware
constraints instead of another global replay target set.

## What followed, in phases

After replay planning was made profile-aware, the work split into clear phases.
The version-by-version detail — every objective name, attempt count, and
acceptance tally — lives in the [Deep research review](./deep-research-review.md)
and the [Research implementation map](./research-implementation-map.md). This
page records the shape of the arc.

| Phase | Mechanics added | Outcome |
| --- | --- | --- |
| Operating system | experiment registry, training recipes, corpus hygiene, candidate quarantine, closed-world verifier, extracted replay planner, constraint-first promotion, transformer module boundaries | built the discipline that lets a screen be audited rather than trusted |
| Anti-collapse objectives | profile target-share pressure, prompt-specific ownership, baseline replay anchors and floor gating, objective-side floor anchors | each screen rejected under the constraint-first gate; coverage and diagnostics advanced, promotion did not |
| Calibrated safe movement | sub-floor learning-rate scales, profile-scale memory, diversity-aware and coverage-frontier acceptance | small numbers of guarded source-profile updates accepted; promotion still blocked |
| Collapsed-profile binding | targeted binding for `learning`, `owner`, and `paraphrases`; protected-learning rejection evidence | narrowed final collapse but did not clear the gate |
| Memory rail and consolidation | corpus-only retrieval memory, memory-consolidation plan, gated source-plan training, profile-specific missing-token pressure | retrieval exact; promotion still blocked |
| Routing diagnostics | root-cause taxonomy, branch routing audit, logit-prior and centroid-separation instrumentation, hidden-projection margin candidate | identified a critical `target_routing_gap` driven by hidden projection; promotion still blocked |

The consistent result across every objective phase is that the transformer
remains rejected for promotion on `branch_diversity_target`. The model collapses
multi-target evaluation profiles to too few predicted branch tokens — it learns
one dominant token instead of routing each prompt to its own answer. The cause
is now measured rather than guessed: representation separation across the `9/9`
multi-target profiles is low, and the dominant-token wins are hidden-projection
driven. See [Branch diversity research](./branch-diversity-research.md) for the
current evidence and the next candidate.

## Two evidence rails, kept apart

The memory rail is the most important thing the consolidation phase produced,
and the easiest to misread. A corpus-only retrieval rail builds memory cards
from the closed corpus and answers `219/219` evaluation probes exactly, with
provenance and no external embeddings, retriever, or weight updates.

That is success for the memory-first rail. It is not neural promotion. The
retrieval rail proves the corpus *contains* every answer; it does not prove the
from-scratch transformer *learned* to produce them. The `memory-served` result
is held separate from the `weight-consolidated` claim, and consolidation from
the memory rail into neural behavior is allowed only when the branch-diversity
and target-token gates pass. They have not. See
[Transformer](../build/transformer.md) for how the two rails sit in the wider
model.

## What this keeps honest

The arc above is mostly a record of rejected screens, and that is the point.
New behavior must be trained from admitted data, measured per profile, and
rejected when it improves one metric by erasing another. A run is kept as
versioned diagnostic evidence whether it is promoted or not, so the audit can
read the failures as readily as the wins. QuarkLM only claims it learned
something new after that admission-and-evidence chain is visible — not because a
run completed, and not because retrieval can already serve the answer.
