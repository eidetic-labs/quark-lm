---
title: Release Candidate Readiness
description: RC tracks, checklists, and non-claims for QuarkLM.
---

# Release Candidate Readiness

<p className="qlm-meta"><span>5 min read</span><span>For contributors</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will learn**

- The three independent tracks a release advances on, and why they must stay separate
- The current tagging decision and the two blockers behind it
- The evidence each decision rests on, and the commands and artifacts a tag requires
- The claims a release is forbidden to make, and the one it is allowed to make

</div>

This page records what QuarkLM may tag, what it may not, and the evidence each
decision rests on. The governing principle is that a release names only what the
project can show: a run is not promoted because it completed, and a track is not
ready because the work is interesting. For the per-run promotion checklist, see
[Release discipline](./release-discipline.md).

## Three tracks, kept separate

QuarkLM advances on three independent tracks. They describe different claims, so
they must not be conflated when tagging or announcing a release.

<div className="qlm-grid">
<div><h4>Alpha</h4><p>Would claim the research scaffold is reproducible and inspectable enough for outside contributors to run and improve, while plainly stating that the language model is not promoted. <strong>Not tag-ready;</strong> blocked by <code>ALPHA_GATE.md</code> Single Responsibility Principle findings.</p></div>
<div><h4>Research Prototype RC</h4><p>Would claim the closed-world self-improvement system is reproducible, auditable, documented, and honest about what is and is not learned into weights. <strong>Deferred</strong> until the alpha quality gates pass.</p></div>
<div><h4>Language Model RC</h4><p>Would claim the from-scratch transformer answers reliably from the admitted corpus without hidden candidate selection and passes neural promotion gates. <strong>Not ready;</strong> blocked on <code>branch_diversity_target</code>.</p></div>
</div>

Alpha gates the scaffold. The Research Prototype RC gates the self-improvement
system as a whole. The Language Model RC gates the neural weights — the
`weight-consolidated` claim — and is the strictest of the three because it is the
one most easily overstated.

## Current decision

`RC_DECISION.md` defers RC tagging. The next possible public tag is the SemVer
alpha described in `ALPHA_GATE.md`, and that tag is blocked until the SRP
findings are resolved.

It is too early to call QuarkLM a release candidate for two reasons:

- the from-scratch transformer is not a promoted working language model; and
- the source tree still contains oversized multi-responsibility modules that the
  alpha gate flags.

<div className="qlm-keypoint">

**A smaller hidden advantage is movement, not a solved gate**

The transformer is not a Language Model RC until branch routing passes. v0.115
lowers average collapsed-token hidden advantage from about `0.0842` to `0.0736`,
but all `9/9` multi-target profiles still collapse to `"n"`. That is movement on
the diagnostic, not a cleared gate.

</div>

## Current evidence

<div className="qlm-grid">
<div><h4>Promoted responder</h4><p><code>runs/self-improve-v0.42/</code> — the current promoted responder run.</p></div>
<div><h4>Latest transformer screen</h4><p><code>runs/transformer-answer-v0.115.0-hidden-projection-margin-candidate-step1-dim4-context80/</code> — passes <code>10/11</code> constraints; still fails <code>branch_diversity_target</code>.</p></div>
</div>

The latest transformer screen is unpromoted. Passing `10/11` constraints is
recorded as diagnostic progress, not as a promotion: a single failing gate is
enough to keep the screen out of the promoted line. The distinction between
serving an answer from memory and consolidating it into weights is the reason a
near-passing screen is still rejected; see
[Language model](../learn/language-model.md) for the three evidence states and
[Transformer](../build/transformer.md) for the branch-diversity blocker.

## Required commands

Run these before tagging or announcing any release:

```bash title="Run the repository and docs checks"
npm run check
```

Before any alpha tag, also run:

```bash title="Run the alpha gate"
npm run alpha:gate
```

The 2026-06-16 verification pass ran the Python suite successfully: `275` tests
passed, the full site build passed, the shared current-state JSON validated, and
`npm run check:release` passed with SRP alpha warnings. That evidence supports
continued development, but the SRP warnings mean the alpha gate itself remains
blocked.

:::note

The local site build validates both public surfaces. Read the Docs publishes the
Docusaurus docs; GitHub Pages publishes only the standalone marketing site. Both
must stay aligned with promoted state.

:::

## Required artifacts

Each track adds artifacts to the one below it. A track is not ready until every
artifact in its column and the columns beneath it is present and current.

<ol className="qlm-steps">
<li><strong>Alpha requires</strong><p><code>ALPHA_GATE.md</code>; SemVer identifier <code>0.115.0-alpha.1</code> with Git tag <code>v0.115.0-alpha.1</code>; <code>npm run alpha:gate</code> passing; README, STATUS, Docusaurus, and marketing current-state alignment.</p></li>
<li><strong>Research Prototype RC additionally requires</strong><p><code>RC_SPEC.md</code>, <code>RC_GAP_AUDIT.md</code>, <code>RC_DECISION.md</code>, <code>RC_CHECKLIST.md</code>, <code>experiment_intent.json</code>, <code>corpus_hygiene.json</code>, <code>training_plan.json</code>, <code>candidate_quarantine.json</code>, <code>closed_world_verifier.json</code>, <code>training_recipe.json</code>, and <code>constraint_first_promotion.json</code>; README, STATUS, Docusaurus, and marketing current-state alignment; and <code>sites/DEPLOYMENT.md</code>, <code>.readthedocs.yaml</code>, and the marketing Pages workflow reviewed for hosting drift.</p></li>
<li><strong>Language Model RC additionally requires</strong><p>passing <code>branch_diversity_target</code>; non-collapsed multi-target branch profiles; target-token coverage floors met; direct-answer evals accepted without hidden candidate selection; and retention and unknown-policy checks passing for the neural learner.</p></li>
</ol>

The Research Prototype RC artifacts are the same closed-world controls the
[closed-world verifier](./closed-world-verifier.md) and
[candidate quarantine](./candidate-quarantine.md) write per run; the RC gate is
where they are reviewed for the release rather than for a single screen.

## Forbidden claims

A release must not claim:

- QuarkLM is a production language model;
- retrieval success is neural weight learning;
- v0.115 solved branch routing;
- the transformer is promoted while `branch_diversity_target` fails;
- the project has proven "world's first" status.

<div className="qlm-keypoint">

**The first two are exactly what track separation prevents**

A production claim overstates the transformer, and treating retrieval success as
weight learning collapses `memory-served` into `weight-consolidated`. Keeping the
tracks separate is what keeps both claims from being made.

</div>

## Allowed current claim

QuarkLM is an experimental closed-world research prototype with a reproducible
admitted-corpus learning loop, exact retrieval and responder evidence, and an
unpromoted from-scratch transformer whose next blocker is branch routing.

## Next model step

When the version loop resumes, prefer the profile-balanced routing repair bundle
from `RC_GAP_AUDIT.md`: target-balanced branch batches across the failing
profiles, hidden-projection margin, representation-separation pressure,
coverage-preserving guards, and branch-diversity acceptance gates. The acceptance
gates are the point of the bundle — a repair counts only when it clears
`branch_diversity_target`, not when it shifts the diagnostic numbers.

Whatever lands, the docs move with the version. If this page references current
eval counts, run ids, commands, or roadmap commitments after the next promoted
run, it is stale until updated; see [Docs drift](./docs-drift.md).

<div className="qlm-next">
<a href="../release-discipline/"><strong>Read next</strong><span>Release discipline</span><small>The per-run promotion checklist and SemVer release-identifier rules.</small></a>
<a href="../../build/transformer/"><strong>The blocker</strong><span>The transformer</span><small>Why the from-scratch model is unpromoted, blocked on branch_diversity_target.</small></a>
<a href="../../learn/language-model/"><strong>Go deeper</strong><span>Language model</span><small>The three evidence states: memory-served is not weight-consolidated.</small></a>
</div>
