---
title: Docs Drift
description: Keep docs and marketing synchronized with releases.
---

# Docs Drift

QuarkLM treats docs as part of the model-improvement system.

The current rule:

- README is a concise repository front door. Durable model philosophy, release
  posture, evidence history, deployment details, and long-form explanations
  belong in Docusaurus and should be linked from README rather than copied
  there.
- README, STATUS, GOAL, and QUALITY are reviewed every promoted version.
- Docusaurus docs are reviewed every promoted version.
- The marketing page is reviewed every promoted version when it references
  current state, evals, commands, domains, or roadmap commitments.
- New release identifiers use SemVer (Semantic Versioning) with
  `vMAJOR.MINOR.PATCH`; the current pre-1.0 line advances as `v0.100.0`,
  `v0.101.0`, `v0.102.0`, and so on. Do not use `XX.YY.ZZ` placeholders in
  release docs. Historical run paths keep their original names for provenance.

The anti-drift mechanism starts with `sites/shared/current-state.json`. Docs and
marketing should read promoted-version facts from that shared source wherever
possible.
