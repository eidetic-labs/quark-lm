---
title: Operate
description: Promote QuarkLM releases with evidence.
slug: /operate/
---

# Operate

Operating QuarkLM means deciding when a run becomes a release and proving that
the decision was earned. A run is not promoted because it completed. It is
promoted only when it preserves the closed-world boundary, passes the recorded
gates, and leaves behind machine-checkable evidence — and when the docs that
describe current state move with it.

The pages under Operate cover three jobs: keeping every run's evidence
auditable, deciding what is eligible to influence the next weight update, and
keeping the public surfaces honest about what is and is not learned.

## The discipline in one rule

Promotion is constraint-first. Closed-world constraints — data boundary,
candidate exclusion, quarantine validity, branch coverage, branch diversity,
target-coverage preservation — must pass *before* any loss, NLL, rank, or
exact-quality number is allowed to count toward promotion. Quality metrics stay
advisory until the constraint report says they are eligible. This is why a run
can produce better numbers and still be rejected, and why the transformer is not
promoted while `branch_diversity_target` fails.

Two distinctions are enforced across every page here:

- A generated lesson, probe, repair, or memory proposal is **not training
  data** until it is admitted to `corpus/ledger.json`. Candidate records carry
  history; the candidate store is not a training source.
- Retrieval answering a probe is **memory-served**, not **weight-consolidated**.
  Exact retrieval evidence proves the corpus contains the answer; it does not
  prove the transformer learned it. See [Build](../build/index.md).

## Operating surfaces

Each promoted run carries a fixed bundle of evidence artifacts. These are JSON
written during the run, validated deterministically, with no hidden promotion
behavior.

| Artifact | Records |
| --- | --- |
| `experiment_intent.json` | Hypothesis, allowed data, planned artifacts, acceptance gates, failure criteria, and the closing decision for a run. |
| `corpus_hygiene.json` | Source mixture, duplicates, train/eval overlap, candidate ratio, and rare-profile coverage. |
| `training_plan.json` | Allowed data sources, the closed-world data boundary, scheduled example mixture, replay summary, and planned artifacts. |
| `candidate_quarantine.json` | Candidate lifecycle state and proof that generated candidates are not training data until admitted. |
| `closed_world_verifier.json` | Deterministic pass/fail evidence for candidate checks and training-plan approval. |
| `training_recipe.json` | Reproducible model, tokenizer, data, objective, optimizer, artifact, and gate recipe. |
| `constraint_first_promotion.json` | The gate that blocks quality metrics until closed-world constraints pass. |
| `corpus_snapshot.json` | Current ledger source hashes and record counts. |
| `corpus_diff.json` | Comparison to the previous promoted run. |

Alongside the per-run artifacts sit the control documents that govern whether a
run is allowed to become a tag, and the hosting controls that keep the public
surfaces in sync.

| Control | Purpose |
| --- | --- |
| `ALPHA_GATE.md` / `RC_SPEC.md` / `RC_GAP_AUDIT.md` / `RC_CHECKLIST.md` | Alpha gate, release-candidate track, gap, checklist, and forbidden-claim controls. |
| `.readthedocs.yaml` / `sites/DEPLOYMENT.md` | Docs-on-Read-the-Docs and marketing-on-GitHub-Pages hosting controls. |
| README / Docusaurus / marketing | Public state that must not drift. |

Run directories are versioned and immutable: `runs/self-improve-*` holds
promoted responder evidence, and unpromoted transformer screens keep their
original names so provenance stays exact. Failed runs are not discarded; they
remain as versioned diagnostic evidence.

## Docs are a promotion gate

Docs are part of the loop, not a byproduct of it. The release is not complete
until the docs are complete: if a page references current eval counts, commands,
run ids, hosting targets, or roadmap commitments, that page must move with the
version. This is an anti-drift discipline, not a training input — promoted-version
facts are read from the shared `sites/shared/current-state.json` source wherever
possible so README, Docusaurus, and marketing cannot quietly disagree.

## Where to read next

| Page | Covers |
| --- | --- |
| [Release readiness](./release-candidate.md) | The alpha and release-candidate tracks, the current decision, required commands and artifacts, and the claims that are forbidden until the evidence supports them. |
| [Release discipline](./release-discipline.md) | The full checklist a promoted run must satisfy, and SemVer release-identifier rules. |
| [Experiment registry](./experiment-registry.md) | Why every run starts with an explicit intent instead of a loose command. |
| [Corpus hygiene](./corpus-hygiene.md) | How data risk is made visible before any metric is interpreted. |
| [Candidate quarantine](./candidate-quarantine.md) | The candidate lifecycle and the rule that keeps generated material out of training until admitted. |
| [Closed-world verifier](./closed-world-verifier.md) | The deterministic check that decides whether a candidate or plan may influence the next learning step. |
| [Training recipes](./training-recipes.md) | Reproducible recipes and the constraint-first promotion report. |
| [Provenance](./provenance.md) | Corpus snapshots and diffs, recorded next to weight and eval changes. |
| [Docs drift](./docs-drift.md) | The rule that keeps docs and marketing synchronized with releases. |
