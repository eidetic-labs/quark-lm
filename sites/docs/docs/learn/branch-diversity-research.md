---
title: Branch Diversity Research
description: External research and v0.115 hidden-projection evidence for QuarkLM branch-diversity failures.
---

# Branch Diversity Research

<p className="qlm-meta"><span>8 min read</span><span>For contributors</span><span>Last reviewed 2026-06-15</span></p>

<div className="qlm-lead">

**What you will learn**

- Why `branch_diversity_target` is the gate that keeps the from-scratch transformer unpromoted
- The run-by-run evidence (v0.112 through v0.115) that narrows the failure
- The external research consulted while choosing each next repair, and what it does and does not license
- The decision standing for the next repair

</div>

Branch diversity is the current transformer bottleneck. The promotion gate
`branch_diversity_target` fails when multi-target evaluation profiles collapse
to too few predicted branch tokens: the model learns to predict one dominant
token rather than routing each prompt to its own answer. Until that gate
passes, the from-scratch transformer is not promoted. See
[Transformer](../build/transformer.md) for how the gate fits the wider model.

This page is the standing record of two things: the run-by-run evidence that
narrows the failure, and the external research consulted while choosing each
next repair. It is the diagnostic companion to the
[transformer screen history](../build/transformer-screen-history.md), which
lists every screen, and to the [forward research plan](./forward-research-plan.md)
and [deep research review](./deep-research-review.md), which frame the strategy.

## The distinction this page protects

Two facts are true at the same time and must not be merged:

- Retrieval memory serves the admitted corpus exactly, answering `219/219`
  evaluation probes with provenance and no weight updates.
- The transformer's neural weights have not learned to route those answers, so
  promotion stays blocked.

The first is `memory-served`; the second is `weight-consolidated`.

<div className="qlm-keypoint">

**Memory-served is not weight-consolidated**

Retrieval answering every probe proves the corpus contains the answer. It does
not prove the transformer learned it, and this page never treats retrieved
answers as learned weights.

</div>

## The diagnostic arc

The recent screens stopped adding new branch objectives and instead diagnosed
why the existing ones collapse. The sections below read newest-first; the
underlying progression is:

```text title="diagnostic-arc.txt"
v0.112  classify the failure         -> critical target_routing_gap
v0.113  audit the routing gap        -> output-bias escape + low separation
v0.114  instrument logit priors      -> hidden-projection pressure, 9/9 profiles
v0.115  first repair candidate       -> hidden advantage down, gate still fails
```

Each screen keeps retrieval exact at `219/219` and remains rejected on
`branch_diversity_target`. None of them is promotion evidence; all of them are
versioned diagnostic evidence for the next repair.

<div className="qlm-keypoint">

**None of these screens is promotion evidence**

A repair that improved a metric by relaxing `branch_diversity_target` would not
count. Every screen below stays rejected on the gate by design — they are
diagnostics that narrow the next repair, not steps toward promotion.

</div>

## v0.115 Evidence

Candidate run:
`runs/transformer-answer-v0.115.0-hidden-projection-margin-candidate-step1-dim4-context80/`.

v0.115 adds `branch-hidden-projection-margin-unlikelihood`, a repair candidate
that compares target-token `hidden * output_weight` contributions without using
output bias as the margin surface. The screen runs one direct-answer step with
output bias frozen. It reduces average collapsed-token hidden advantage from
about `0.0842` to `0.0736`, which supports hidden projection as a relevant
repair surface.

The candidate is still rejected for neural promotion. Constraint-first
promotion passes `10/11` constraints and fails `branch_diversity_target`; all
`9/9` multi-target profiles still collapse to `"n"`, `2` profiles keep zero
target-token coverage, and hidden-projection pressure remains primary across
`9/9` profiles. The next repair must scale beyond a single branch batch while
preserving coverage and representation-separation gates.

## v0.114 Evidence

Diagnostic run:
`runs/transformer-answer-v0.114.0-logit-prior-representation-instrumentation-profile-specific-memory-consolidation-step1-dim4-context80/`.

The run consumes the v0.113 memory-consolidation plan, targets `owner`,
`paraphrases`, and `glossary`, keeps retrieval exact at `219/219`, records
`24` profile-specific missing-token attempts with `0` direct missing-token
acceptances and `8` fallbacks, and remains rejected on
`branch_diversity_target`.

The root-cause diagnosis remains a critical `target_routing_gap`. The routing
audit still flags high output-bias escape risk (`"n"` bias rank `1`, `"a"` bias
rank `3`), but `branch_logit_prior_profiles` show the dominant-token wins are
driven by hidden-projection pressure across `9/9` multi-target profiles.
Centroid separation remains poor across the sampled profiles.

## v0.113 Evidence

Diagnostic run:
`runs/transformer-answer-v0.113.0-branch-routing-audit-profile-specific-memory-consolidation-step1-dim4-context80/`.

The run consumes the v0.112 memory-consolidation plan, targets `owner`,
`paraphrases`, and `learning`, keeps retrieval exact at `219/219`, records
`18` profile-specific missing-token attempts with `0` direct missing-token
acceptances and `6` fallbacks, and remains rejected on
`branch_diversity_target`.

The root-cause diagnosis remains a critical `target_routing_gap`: `9/9`
profiles fail, `3` remain collapsed, `1` has zero target-token coverage, and
`6` have buried targets. The new `branch_routing_audit` narrows the next
mechanics target:

- `audit_hypothesis`: `routing_gap_requires_representation_and_logit_audit`
- output-bias escape risk: `high`, with `"n"` at bias rank `2`
- prompt-to-branch representation separation: low across `9/9` multi-target
  profiles
- minimum different-target hidden distance: about `0.00077`
- target imbalance hotspot: `glossary`, with top target share `0.6316`

## v0.112 Evidence

Diagnostic run:
`runs/transformer-answer-v0.112.0-branch-diversity-root-cause-profile-specific-memory-consolidation-step1-dim4-context80/`.

The run consumes the v0.111 memory-consolidation plan, targets `owner`,
`paraphrases`, and `glossary`, keeps retrieval exact at `219/219`, records
`24` profile-specific missing-token attempts with `0` direct missing-token
acceptances and `8` fallbacks, and remains rejected on
`branch_diversity_target`.

The new root-cause diagnostic classifies the final failure as
`target_routing_gap` with `critical` severity. It records `9/9` failed
profiles, `3` collapsed profiles, `1` zero-coverage profile, `6` buried-target
profiles, and reused dominant tokens: `"n"` across `5` profiles and `"a"`
across `4` profiles. The worst profile is `paraphrases`: `0.0` target-token
coverage, `predicted_unique: 1`, and average target rank `22.5`.

## What External Work Suggests

The sources below are design references consulted while choosing each repair.
None of them is promotion evidence; each row records what other work does and
the narrower implication for QuarkLM's closed-world branch gate.

| Source | What others do | QuarkLM implication |
| --- | --- | --- |
| [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) | Shows that common decoding choices can produce bland or repetitive language even from strong likelihood-trained models. | Decoding diversity is not enough. QuarkLM needs branch diversity in the learned distribution before promotion. |
| [Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319) | Penalizes undesirable tokens or sequences during training. | QuarkLM's unlikelihood variants can move the collapse token, but v0.112 says routing remains broken. |
| [Hugging Face generation configuration](https://huggingface.co/docs/transformers/en/main_classes/text_generation) | Exposes repetition penalties, no-repeat n-gram controls, diversity penalties, sampling, beam settings, and other generation-time controls. | Sampling or penalties may become inference rails, but they cannot prove closed-world weight consolidation. |
| [Hugging Face generation utilities](https://huggingface.co/docs/transformers/en/internal/generation_utils) | Exposes logits processors, processed score tensors, and optional hidden-state outputs for instrumentation. | v0.114 follows this diagnostic pattern by inspecting output-bias ranks, hidden projections, and prompt-to-branch hidden-state separation. |
| [Diverse Beam Search](https://arxiv.org/abs/1610.02424) | Adds diversity to beam decoding to avoid near-duplicate candidate outputs. | Useful later for candidate exploration; not a substitute for target-token coverage. |
| [fairseq search mechanics](https://github.com/facebookresearch/fairseq/blob/main/fairseq/search.py) | Implements search variants, including diversity-aware beam scoring. | Mature stacks keep search diversity separate from model learning, so QuarkLM should keep decoding diversity out of promotion claims. |
| [Class-Balanced Loss](https://arxiv.org/abs/1901.05555) | Reweights long-tailed classes using effective sample counts. | Audit profile/target imbalance before changing another objective. |
| [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) | Separates representations by label. | Measure whether prompt states separate by branch target before adding output-head pressure. |
| [OLMo](https://arxiv.org/abs/2402.00838) and [LLM360](https://arxiv.org/abs/2312.06550) | Release training code, data, checkpoints, evaluations, and intermediate artifacts. | Keep root-cause diagnostics and promotion decisions artifacted. |
| [nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py) and [minGPT](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py) | Use clean GPT mechanics, cross-entropy training, logits, validation loss, checkpointing, and sampling. | Structure references only; they do not directly solve QuarkLM's tiny closed-world branch gate. |

:::note
These references shape *how* the failure is instrumented and repaired. They do
not change the gate: a screen still has to pass `branch_diversity_target` on
QuarkLM's own closed-world corpus before the transformer is promoted.
:::

## Taxonomy

v0.112 adds `branch_diversity_target.root_cause`, v0.113 adds
`branch_routing_audit`, and v0.114 adds `branch_logit_prior_profiles`. The
root-cause classifier names the failure mode with one of these hypotheses:

<div className="qlm-grid">
<div><h4><code>global_output_prior_collapse</code></h4><p>Multi-target profiles collapse to one shared dominant token.</p></div>
<div><h4><code>profile_local_prediction_collapse</code></h4><p>Profiles collapse, but not to one shared token.</p></div>
<div><h4><code>target_routing_gap</code></h4><p>At least one profile has zero target-token coverage.</p></div>
<div><h4><code>target_rank_burial</code></h4><p>Correct targets are usually outside the top-k set.</p></div>
<div><h4><code>wrong_diversity_not_target_coverage</code></h4><p>Predictions are diverse but miss the target tokens.</p></div>
<div><h4><code>mixed_branch_diversity_gap</code></h4><p>Multiple weaker failure modes appear together.</p></div>
</div>

## Decision

The next repair continues to instrument the route from prompt evidence to
branch target before adding another branch objective. It must hold the existing
gates fixed: a repair that improved a metric by relaxing
`branch_diversity_target` would not count.

<ol className="qlm-steps">
<li><strong>Target hidden-projection contributions</strong><p>Address the contributions that make dominant tokens beat missing target tokens.</p></li>
<li><strong>Compare hidden-state separation</strong><p>Measure prompt-to-branch hidden-state separation for failed profiles.</p></li>
<li><strong>Separate the failure modes</strong><p>Split zero-coverage profiles from buried-target profiles.</p></li>
<li><strong>Add construction and sampling diagnostics</strong><p>Instrument candidate construction and sampling for profile/target imbalance.</p></li>
<li><strong>Require both audits to improve</strong><p>Both <code>branch_diversity_target.root_cause</code> and <code>branch_routing_audit</code> must improve without relaxing <code>branch_diversity_target</code>.</p></li>
</ol>

The screen that acts on this decision is recorded in the
[transformer screen history](../build/transformer-screen-history.md) when it
runs. Until a screen passes the gate, the transformer stays unpromoted and
retrieval memory remains the answering rail.

<div className="qlm-next">
<a href="../build/transformer.md"><strong>Read next</strong><span>The transformer</span><small>How the from-scratch model works and where the gate fits.</small></a>
<a href="../build/transformer-screen-history.md"><strong>Read next</strong><span>Transformer screen history</span><small>Every screen, objective name, run path, and evidence table.</small></a>
<a href="./forward-research-plan.md"><strong>Read next</strong><span>Forward research plan</span><small>The research-backed sequence that frames the strategy.</small></a>
</div>
