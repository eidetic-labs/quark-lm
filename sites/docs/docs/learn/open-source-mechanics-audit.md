---
title: Open-Source Mechanics Audit
description: What QuarkLM learns from open-source LLM and continual-learning mechanics without copying code or data.
---

# Open-Source Mechanics Audit

<p className="qlm-meta"><span>6 min read</span><span>For contributors</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will learn**

- Why QuarkLM studies open-source training mechanics as a *shape* reference and refuses to import anything they learned.
- What the audit asked for, and the one trainer change it named: profile-aware replay planning.
- Why the arc that followed is mostly a record of rejected screens, and why that is the point.
- How the memory rail and the weight rail are kept apart so retrieval success is never read as neural promotion.

</div>

QuarkLM reads open-source projects and papers as design references for how a
training and continual-learning system should be *shaped*. It does not take
model weights, tokenizers, data, embeddings, or copied implementations from any
of them. The full audit lives in the repository root at `MECHANICS_AUDIT.md`;
this page is the durable summary.

The distinction is the same one drawn in [Purity boundary](../secure/purity-boundary.md):
studying the structure of an open model is allowed, importing anything it
learned is not. A reference can suggest where a trainer boundary should sit. It
cannot become a vocabulary, a checkpoint, or a line of training text.

<div className="qlm-keypoint">

**A reference is a shape, not a source**

Studying the structure of an open model is allowed; importing anything it learned is not. A reference can suggest where a trainer boundary should sit — it cannot become a vocabulary, a checkpoint, or a line of training text.

</div>

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

Each card records what QuarkLM studies and, just as importantly, what it refuses
to take across the boundary. The "does not take" line is the load-bearing one.

<div className="qlm-grid">
<div><h4>nanoGPT and minGPT</h4><p><strong>Studies:</strong> compact trainer/model boundaries, checkpoint cadence, optimizer state, generation traces. <strong>Does not take:</strong> code, weights, GPT-2 imports, datasets, tokenizer state.</p></div>
<div><h4>LitGPT</h4><p><strong>Studies:</strong> config-driven decoder-only recipes, norm/rotary/KV-cache mechanics. <strong>Does not take:</strong> implementation code, recipes as training data, model weights.</p></div>
<div><h4>Hugging Face tokenizers</h4><p><strong>Studies:</strong> tokenizer pipeline concepts, special-token and alignment artifacts. <strong>Does not take:</strong> pretrained vocabularies or merge tables.</p></div>
<div><h4>Avalanche</h4><p><strong>Studies:</strong> continual-learning streams, replay strategies, evaluation plugins. <strong>Does not take:</strong> library dependency, benchmark data, external pretrained models.</p></div>
<div><h4>Self-Instruct, STaR, Reflexion</h4><p><strong>Studies:</strong> candidate generation, filtering, and memory-before-weight-update separation. <strong>Does not take:</strong> external-model generated training material.</p></div>
<div><h4>LLM360, OLMo, OLMo 2</h4><p><strong>Studies:</strong> transparent code/data/checkpoint/log/recipe practice and data-mixture reporting. <strong>Does not take:</strong> open training corpora, weights, or external checkpoints.</p></div>
</div>

Self-generated material in particular is not training data until it is verified
against admitted sources and admitted to the ledger; see
[Candidate quarantine](../operate/candidate-quarantine.md).

## What the audit asked for

The audit named a concrete trainer change: replay planning had to become
profile-aware. The mechanics it required were:

<ol className="qlm-steps">
<li><strong>Carry profile keys through branch records</strong><p>Profile identity travels with every branch record instead of being recovered later.</p></li>
<li><strong>Compute missing targets per profile</strong><p>Deficits are measured per profile instead of globally.</p></li>
<li><strong>Preserve represented coverage per profile</strong><p>Coverage that already exists is protected profile by profile.</p></li>
<li><strong>Record a replay-plan artifact</strong><p>One artifact captures profile counts, target sets, represented targets, deficits, and coverage floors.</p></li>
<li><strong>Verify profiles cannot mask each other</strong><p>Focused tests confirm one profile's improvement cannot hide another profile's deficit.</p></li>
</ol>

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

<div className="qlm-grid">
<div><h4>Operating system</h4><p><strong>Mechanics:</strong> experiment registry, training recipes, corpus hygiene, candidate quarantine, closed-world verifier, extracted replay planner, constraint-first promotion, transformer module boundaries. <strong>Outcome:</strong> built the discipline that lets a screen be audited rather than trusted.</p></div>
<div><h4>Anti-collapse objectives</h4><p><strong>Mechanics:</strong> profile target-share pressure, prompt-specific ownership, baseline replay anchors and floor gating, objective-side floor anchors. <strong>Outcome:</strong> each screen rejected under the constraint-first gate; coverage and diagnostics advanced, promotion did not.</p></div>
<div><h4>Calibrated safe movement</h4><p><strong>Mechanics:</strong> sub-floor learning-rate scales, profile-scale memory, diversity-aware and coverage-frontier acceptance. <strong>Outcome:</strong> small numbers of guarded source-profile updates accepted; promotion still blocked.</p></div>
<div><h4>Collapsed-profile binding</h4><p><strong>Mechanics:</strong> targeted binding for <code>learning</code>, <code>owner</code>, and <code>paraphrases</code>; protected-learning rejection evidence. <strong>Outcome:</strong> narrowed final collapse but did not clear the gate.</p></div>
<div><h4>Memory rail and consolidation</h4><p><strong>Mechanics:</strong> corpus-only retrieval memory, memory-consolidation plan, gated source-plan training, profile-specific missing-token pressure. <strong>Outcome:</strong> retrieval exact; promotion still blocked.</p></div>
<div><h4>Routing diagnostics</h4><p><strong>Mechanics:</strong> root-cause taxonomy, branch routing audit, logit-prior and centroid-separation instrumentation, hidden-projection margin candidate. <strong>Outcome:</strong> identified a critical <code>target_routing_gap</code> driven by hidden projection; promotion still blocked.</p></div>
</div>

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

<div className="qlm-keypoint">

**Memory-served is not weight-consolidated**

The retrieval rail proves the corpus *contains* every answer; it does not prove the from-scratch transformer *learned* to produce them. The `memory-served` result is held separate from the `weight-consolidated` claim.

</div>

That is success for the memory-first rail. It is not neural promotion.
Consolidation from the memory rail into neural behavior is allowed only when the
branch-diversity and target-token gates pass. They have not. See
[Transformer](../build/transformer.md) for how the two rails sit in the wider
model.

## What this keeps honest

The arc above is mostly a record of rejected screens, and that is the point.
New behavior must be trained from admitted data, measured per profile, and
rejected when it improves one metric by erasing another. A run is kept as
versioned diagnostic evidence whether it is promoted or not, so the audit can
read the failures as readily as the wins.

:::note
QuarkLM only claims it learned something new after the admission-and-evidence
chain is visible — not because a run completed, and not because retrieval can
already serve the answer.
:::

<div className="qlm-next">
<a href="../branch-diversity-research/"><strong>Read next</strong><span>Branch diversity research</span><small>The current evidence behind the rejected promotion and the next candidate.</small></a>
<a href="../../build/transformer/"><strong>Go deeper</strong><span>The transformer</span><small>How the memory rail and the weight rail sit in the wider model.</small></a>
<a href="../deep-research-review/"><strong>See the detail</strong><span>Deep research review</span><small>Every objective name, attempt count, and acceptance tally behind the arc.</small></a>
</div>
