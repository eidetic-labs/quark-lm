---
title: Operate
description: Promote QuarkLM releases with evidence.
slug: /operate/
---

# Operate

Operating QuarkLM means keeping the model, corpus, evals, reports, docs, and
public surfaces synchronized.

## Operating Surfaces

| Surface | Purpose |
| --- | --- |
| `runs/self-improve-*` | Versioned training and audit evidence. |
| `experiment_intent.json` | Hypothesis, allowed data, planned artifacts, gates, and decision for a run. |
| `corpus_hygiene.json` | Source mixture, duplicates, train/eval overlap, candidate ratio, and rare-profile coverage. |
| `training_plan.json` | Allowed data sources, scheduled example mixture, replay summary, and planned artifacts. |
| `candidate_quarantine.json` | Candidate lifecycle state and proof that generated candidates are not training data until admitted. |
| `closed_world_verifier.json` | Deterministic pass/fail evidence for candidate checks and training-plan approval. |
| `training_recipe.json` | Reproducible model, data, objective, optimizer, artifact, and gate recipe. |
| `constraint_first_promotion.json` | Promotion gate that blocks quality metrics until closed-world constraints pass. |
| `corpus_snapshot.json` | Current ledger source hashes and record counts. |
| `corpus_diff.json` | Comparison to the previous promoted run. |
| `RC_SPEC.md` / `RC_GAP_AUDIT.md` / `RC_CHECKLIST.md` | Release-candidate track, gap, checklist, and forbidden-claim controls. |
| `.readthedocs.yaml` / `sites/DEPLOYMENT.md` | Docs-on-Read-the-Docs and marketing-on-GitHub-Pages hosting controls. |
| README / Docusaurus / marketing | Public state that must not drift. |

Start with [release candidate readiness](./release-candidate.md), then read
[release discipline](./release-discipline.md),
[experiment registry](./experiment-registry.md), [corpus hygiene](./corpus-hygiene.md),
[candidate quarantine](./candidate-quarantine.md),
[closed-world verifier](./closed-world-verifier.md),
[training recipes](./training-recipes.md), [provenance](./provenance.md), and
[docs drift](./docs-drift.md).
