---
title: Release Candidate Readiness
description: RC tracks, checklists, and non-claims for QuarkLM.
---

# Release Candidate Readiness

This page records what QuarkLM may tag, what it may not, and the evidence each
decision rests on. The governing principle is that a release names only what the
project can show: a run is not promoted because it completed, and a track is not
ready because the work is interesting. For the per-run promotion checklist, see
[Release discipline](./release-discipline.md).

## Three tracks, kept separate

QuarkLM advances on three independent tracks. They describe different claims, so
they must not be conflated when tagging or announcing a release.

| Track | What it would claim | Current posture |
| --- | --- | --- |
| Alpha | The research scaffold is reproducible and inspectable enough for outside contributors to run and improve, while plainly stating that the language model is not promoted. | Not tag-ready; blocked by `ALPHA_GATE.md` Single Responsibility Principle findings. |
| Research Prototype RC | The closed-world self-improvement system is reproducible, auditable, documented, and honest about what is and is not learned into weights. | Deferred until the alpha quality gates pass. |
| Language Model RC | The from-scratch transformer answers reliably from the admitted corpus without hidden candidate selection and passes neural promotion gates. | Not ready; blocked on `branch_diversity_target`. |

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

The transformer is not a Language Model RC until branch routing passes. v0.115
lowers average collapsed-token hidden advantage from about `0.0842` to `0.0736`,
but all `9/9` multi-target profiles still collapse to `"n"`. A smaller hidden
advantage is movement on the diagnostic, not a solved gate.

## Current evidence

| Surface | Evidence | What it shows |
| --- | --- | --- |
| Promoted responder | `runs/self-improve-v0.42/` | The current promoted responder run. |
| Latest transformer screen | `runs/transformer-answer-v0.115.0-hidden-projection-margin-candidate-step1-dim4-context80/` | Passes `10/11` constraints; still fails `branch_diversity_target`. |

The latest transformer screen is unpromoted. Passing `10/11` constraints is
recorded as diagnostic progress, not as a promotion: a single failing gate is
enough to keep the screen out of the promoted line. The distinction between
serving an answer from memory and consolidating it into weights is the reason a
near-passing screen is still rejected; see
[Language model](../learn/language-model.md) for the three evidence states and
[Transformer](../build/transformer.md) for the branch-diversity blocker.

## Required commands

Run these before tagging or announcing any release:

```bash
npm run check
```

Before any alpha tag, also run:

```bash
npm run alpha:gate
```

The 2026-06-16 verification pass ran the Python suite successfully: `275` tests
passed, the full site build passed, the shared current-state JSON validated, and
`npm run check:release` passed with SRP alpha warnings. That evidence supports
continued development, but the SRP warnings mean the alpha gate itself remains
blocked.

The local site build validates both public surfaces. Read the Docs publishes the
Docusaurus docs; GitHub Pages publishes only the standalone marketing site. Both
must stay aligned with promoted state.

## Required artifacts

Each track adds artifacts to the one below it. A track is not ready until every
artifact in its column and the columns beneath it is present and current.

**Alpha requires:**

- `ALPHA_GATE.md`
- SemVer identifier `0.115.0-alpha.1` with Git tag `v0.115.0-alpha.1`
- `npm run alpha:gate` passing
- README, STATUS, Docusaurus, and marketing current-state alignment

**Research Prototype RC additionally requires:**

- `RC_SPEC.md`
- `RC_GAP_AUDIT.md`
- `RC_DECISION.md`
- `RC_CHECKLIST.md`
- `experiment_intent.json`
- `corpus_hygiene.json`
- `training_plan.json`
- `candidate_quarantine.json`
- `closed_world_verifier.json`
- `training_recipe.json`
- `constraint_first_promotion.json`
- README, STATUS, Docusaurus, and marketing current-state alignment
- `sites/DEPLOYMENT.md`, `.readthedocs.yaml`, and the marketing Pages workflow
  reviewed for hosting drift

**Language Model RC additionally requires:**

- passing `branch_diversity_target`
- non-collapsed multi-target branch profiles
- target-token coverage floors met
- direct-answer evals accepted without hidden candidate selection
- retention and unknown-policy checks passing for the neural learner

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

The first two are the failures the track separation exists to prevent: a
production claim overstates the transformer, and treating retrieval success as
weight learning collapses `memory-served` into `weight-consolidated`.

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
