---
title: Docs Drift
description: Keep docs and marketing synchronized with releases.
---

# Docs Drift

<p className="qlm-meta"><span>5 min read</span><span>For contributors</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will learn**

- What docs drift is, and why QuarkLM closes it at promotion rather than afterward.
- Why docs are a promotion gate and an anti-drift discipline, not a training input.
- The surfaces reviewed every promoted version, and what a reviewer checks on each.
- How a single shared source of truth keeps README, Docusaurus, and marketing from disagreeing.

</div>

Docs drift is the gap that opens when the prose describing QuarkLM stops
matching the run it describes: an eval count that is one release stale, a
command that was renamed, a roadmap line that promises something the evidence
does not yet show. QuarkLM treats closing that gap as part of promotion, not as
cleanup that happens afterward.

<div className="qlm-keypoint">

**Docs are a promotion gate, not a training input.**

Reviewing and correcting a doc never changes neural weights and never adds
anything to the corpus; the only path that changes weights runs through
admission and the gates, described in [Build](../build/index.md). Docs are
checked so the public surfaces stay honest about what is and is not learned, not
so the model can learn from them.

</div>

## Why docs are gated

A run is promoted only when it preserves the closed-world boundary, passes the
recorded gates, and leaves behind machine-checkable evidence. A release whose
docs still describe the prior state fails the last of those conditions: the
evidence on disk and the prose a reader sees no longer agree. The release is not
complete until the docs are complete.

The risk is specific. Several QuarkLM distinctions are easy to erode in prose
written in a hurry:

<div className="qlm-grid">
<div><h4>memory-served, not weight-consolidated</h4><p>Retrieval answering a probe is <code>memory-served</code>. Exact retrieval proves the corpus contains the answer, not that the transformer learned it.</p></div>
<div><h4>transformer not promoted</h4><p>The from-scratch transformer is not promoted while it is blocked on <code>branch_diversity_target</code>.</p></div>
<div><h4>candidates are not training data</h4><p>Generated lessons, probes, and repairs are not training data until they are admitted to <code>corpus/ledger.json</code>.</p></div>
</div>

Each of these is a claim a stale or careless doc can quietly overstate. Gating
docs at promotion is how those overstatements are caught before a reader sees
them.

## Surfaces reviewed every promoted version

<div className="qlm-grid">
<div><h4>README</h4><p>Concise repository front door. Links to the durable material rather than duplicating it. Reviewed every promoted version.</p></div>
<div><h4>STATUS / GOAL / QUALITY</h4><p>Current state, intent, and quality posture in the repository. Reviewed every promoted version.</p></div>
<div><h4>Docusaurus docs</h4><p>The durable model philosophy, release posture, evidence history, deployment detail, and long-form explanation. Reviewed every promoted version.</p></div>
<div><h4>Marketing page</h4><p>Public positioning. Reviewed every promoted version that touches current state, evals, commands, domains, or roadmap commitments.</p></div>
<div><h4>ALPHA_GATE.md</h4><p>Alpha readiness, single-responsibility blockers, and release-tag guidance. Reviewed whenever alpha readiness, an SRP blocker, or tag guidance changes.</p></div>
</div>

The README is deliberately thin. Durable model philosophy, release posture,
evidence history, deployment details, and long-form explanation belong in
Docusaurus and are linked from the README rather than copied into it, so a fact
has one home and one place to update.

## The shared source of truth

The anti-drift mechanism starts with a single shared file,
`sites/shared/current-state.json`. It holds the promoted-version facts that
otherwise tend to be retyped and fall out of sync: product and domain names, the
proposed alpha version and tag, the current alpha status, the release-candidate
decision, and the recorded status of the from-scratch transformer.

Docs and marketing read those facts from the shared source wherever possible
instead of restating them inline.

```text title="One home, three surfaces"
                 sites/shared/current-state.json
                 (promoted-version facts: one home)
                  /            |             \
              README       Docusaurus      marketing
```

Because all three surfaces draw from the same file, they cannot quietly
disagree about the current version, the alpha status, or what the transformer
has and has not demonstrated. A single edit to the shared source propagates
rather than leaving stragglers behind.

## What a reviewer checks

A page must move with the version when it references anything the version can
change. If a page touches none of these, it can be left as it stands. If it
touches any, it is reviewed against the run before the version is called
complete.

<div className="qlm-grid">
<div><h4>Current eval counts</h4><p>Re-check: a promoted run can change them.</p></div>
<div><h4>Commands or CLI invocations</h4><p>Re-check: flags and module names are renamed across versions.</p></div>
<div><h4>Run ids or run paths</h4><p>Re-check: new runs are added; historical paths keep their original names for provenance.</p></div>
<div><h4>Hosting targets or domains</h4><p>Re-check: deployment configuration can move.</p></div>
<div><h4>Roadmap commitments</h4><p>Re-check: promised work must not outrun the evidence that supports it.</p></div>
</div>

## Version identifiers

Docs that name a release identifier follow the SemVer rules recorded in
[Release discipline](./release-discipline.md): `MAJOR.MINOR.PATCH` with optional
prerelease labels such as `0.115.0-alpha.1`, a conventional `v` prefix on Git
tags, no `XX.YY.ZZ` placeholders in release docs, and historical run paths kept
under their original names so provenance stays exact.

## Rule

:::tip

Docs are part of the model-improvement system, reviewed as a promotion gate, but
they are never training input. A promoted version that changes current state,
evals, commands, domains, or roadmap commitments is not complete until every
surface that references those facts has moved with it, reading from
`sites/shared/current-state.json` wherever it can.

:::

<div className="qlm-next">
<a href="../release-discipline/"><strong>Read next</strong><span>Release discipline</span><small>The promotion gates a run clears, and the version-naming rules docs follow.</small></a>
<a href="../../build/"><strong>Reference</strong><span>Build</span><small>The only path that changes weights: admission and the gates, where docs are not training input.</small></a>
<a href="../"><strong>Back to</strong><span>Operating surfaces</span><small>The full set of run artifacts and disciplines that gate a promotion.</small></a>
</div>
