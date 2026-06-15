---
title: Docs Drift
description: Keep docs and marketing synchronized with releases.
---

# Docs Drift

QuarkLM treats docs as part of the model-improvement system.

The current rule:

- README, STATUS, GOAL, and QUALITY are reviewed every promoted version.
- Docusaurus docs are reviewed every promoted version.
- The marketing page is reviewed every promoted version when it references
  current state, evals, commands, domains, or roadmap commitments.
- New release identifiers use SemVer-style `vMAJOR.MINOR.PATCH`; historical
  run paths keep their original names for provenance.

The anti-drift mechanism starts with `sites/shared/current-state.json`. Docs and
marketing should read promoted-version facts from that shared source wherever
possible.
