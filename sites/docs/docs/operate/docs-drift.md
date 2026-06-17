---
title: Docs Drift
description: Keep docs and marketing synchronized with releases.
---

# Docs Drift

Docs drift is the gap that opens when the prose describing QuarkLM stops
matching the run it describes: an eval count that is one release stale, a
command that was renamed, a roadmap line that promises something the evidence
does not yet show. QuarkLM treats closing that gap as part of promotion, not as
cleanup that happens afterward.

This is a promotion gate and an anti-drift discipline. It is **not** a training
input. Reviewing and correcting a doc never changes neural weights and never
adds anything to the corpus; the only path that changes weights runs through
admission and the gates, described in [Build](../build/index.md). Docs are
checked so the public surfaces stay honest about what is and is not learned, not
so the model can learn from them.

## Why docs are gated

A run is promoted only when it preserves the closed-world boundary, passes the
recorded gates, and leaves behind machine-checkable evidence. A release whose
docs still describe the prior state fails the last of those conditions: the
evidence on disk and the prose a reader sees no longer agree. The release is not
complete until the docs are complete.

The risk is specific. Several QuarkLM distinctions are easy to erode in prose
written in a hurry:

- that retrieval answering a probe is `memory-served`, not
  `weight-consolidated` — exact retrieval proves the corpus contains the answer,
  not that the transformer learned it;
- that the from-scratch transformer is not promoted while it is blocked on
  `branch_diversity_target`;
- that generated lessons, probes, and repairs are not training data until they
  are admitted to `corpus/ledger.json`.

Each of these is a claim a stale or careless doc can quietly overstate. Gating
docs at promotion is how those overstatements are caught before a reader sees
them.

## Surfaces reviewed every promoted version

| Surface | Role | Reviewed when |
| --- | --- | --- |
| `README` | Concise repository front door. Links to the durable material rather than duplicating it. | Every promoted version. |
| `STATUS` / `GOAL` / `QUALITY` | Current state, intent, and quality posture in the repository. | Every promoted version. |
| Docusaurus docs | The durable model philosophy, release posture, evidence history, deployment detail, and long-form explanation. | Every promoted version. |
| Marketing page | Public positioning. | Every promoted version that touches current state, evals, commands, domains, or roadmap commitments. |
| `ALPHA_GATE.md` | Alpha readiness, single-responsibility blockers, and release-tag guidance. | Whenever alpha readiness, an SRP blocker, or tag guidance changes. |

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

```text
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
change:

| The page mentions… | Then it must be re-checked because… |
| --- | --- |
| Current eval counts | A promoted run can change them. |
| Commands or CLI invocations | Flags and module names are renamed across versions. |
| Run ids or run paths | New runs are added; historical paths keep their original names for provenance. |
| Hosting targets or domains | Deployment configuration can move. |
| Roadmap commitments | Promised work must not outrun the evidence that supports it. |

If a page touches none of these, it can be left as it stands. If it touches any,
it is reviewed against the run before the version is called complete.

## Version identifiers

Docs that name a release identifier follow the SemVer rules recorded in
[Release discipline](./release-discipline.md): `MAJOR.MINOR.PATCH` with optional
prerelease labels such as `0.115.0-alpha.1`, a conventional `v` prefix on Git
tags, no `XX.YY.ZZ` placeholders in release docs, and historical run paths kept
under their original names so provenance stays exact.

## Rule

Docs are part of the model-improvement system, reviewed as a promotion gate, but
they are never training input. A promoted version that changes current state,
evals, commands, domains, or roadmap commitments is not complete until every
surface that references those facts has moved with it, reading from
`sites/shared/current-state.json` wherever it can.
