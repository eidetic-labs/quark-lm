---
title: Release Candidate Readiness
description: RC tracks, checklists, and non-claims for QuarkLM.
---

# Release Candidate Readiness

QuarkLM has two release-candidate tracks. They must stay separate.

| Track | Meaning | Current posture |
| --- | --- | --- |
| Research Prototype RC | The closed-world self-improvement system is reproducible, auditable, documented, and honest about what is and is not learned into weights. | Near. |
| Language Model RC | The from-scratch transformer reliably answers from the admitted corpus without candidate crutches and passes neural promotion gates. | Not ready. |

The current promoted responder evidence is `runs/self-improve-v0.42/`. The
latest unpromoted transformer screen is
`runs/transformer-answer-v0.115.0-hidden-projection-margin-candidate-step1-dim4-context80/`.
That screen passes `10/11` constraints but still fails
`branch_diversity_target`.

## Current Decision

Pursue **Research Prototype RC** first. It is the honest near-term release
candidate because the system already has corpus boundaries, deterministic
verifier checks, candidate quarantine, recipes, constraint-first promotion,
docs discipline, and rejected transformer evidence.

Do not call the transformer a Language Model RC until branch routing passes.
v0.115 lowers average collapsed-token hidden advantage from about `0.0842` to
`0.0736`, but all `9/9` multi-target profiles still collapse to `"n"`.

## Required Commands

Run these before tagging or announcing an RC:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
npm run sites:build
python3 -m json.tool sites/shared/current-state.json >/dev/null
```

The local site build validates both public surfaces. Read the Docs publishes
the Docusaurus docs, and GitHub Pages publishes only the standalone marketing
site.

## Required Artifacts

Research Prototype RC requires:

- `RC_SPEC.md`
- `RC_GAP_AUDIT.md`
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

Language Model RC additionally requires:

- passing `branch_diversity_target`
- non-collapsed multi-target branch profiles
- target-token coverage floors met
- direct-answer evals accepted without hidden candidate selection
- retention and unknown-policy checks passing for the neural learner

## Forbidden Claims

Do not claim:

- QuarkLM is a production language model
- retrieval success is neural weight learning
- v0.115 solved branch routing
- the transformer is promoted while `branch_diversity_target` fails
- the project has proven "world's first" status

Allowed current claim:

QuarkLM is an experimental closed-world research prototype with a reproducible
admitted-corpus learning loop, exact retrieval/responder evidence, and an
unpromoted from-scratch transformer whose next blocker is branch routing.

## Next Model Step

When the version loop resumes, prefer the profile-balanced routing repair bundle
from `RC_GAP_AUDIT.md`: target-balanced branch batches across failing profiles,
hidden-projection margin, representation-separation pressure, coverage-preserving
guards, and branch-diversity acceptance gates.
