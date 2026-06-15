---
title: Corpus Hygiene
description: Corpus hygiene and training-plan artifacts for QuarkLM runs.
---

# Corpus Hygiene

v0.73 adds `src/closed_world_lm/corpus_hygiene.py` and two required run
artifacts:

- `corpus_hygiene.json`
- `training_plan.json`

Self-improvement answer cycles and transformer answer-training runs write these
artifacts before their metrics are treated as evidence.

## Corpus Hygiene Report

`corpus_hygiene.json` records:

- corpus source counts;
- training text path and character count;
- training-example source mixture;
- duplicate training examples;
- duplicate admission and eval ids;
- train/eval prompt overlap;
- protected heldout prompt overlap;
- candidate-example ratio;
- rare-profile coverage.

The report does not promote or reject a model by itself. It makes data risk
visible before promotion gates or transformer screens interpret metrics.

## Training Plan

`training_plan.json` records:

- component and run id;
- allowed data sources;
- closed-world data boundary;
- hygiene report path;
- eval-set counts;
- base and scheduled training-example mixture;
- candidate policy status;
- training recipe path and summary when v0.77 recipes are written;
- replay-plan path and summary when profile-aware replay writes one;
- closed-world verifier path and summary when v0.76 approval is written;
- planned artifacts.

Candidate ratio is reported in v0.73. v0.75 adds
`candidate_quarantine.json` and links it from `training_plan.json`, so generated
or proposed examples still must not become training data without a later
admission and verification path. v0.76 adds `closed_world_verifier.json`, so
the training plan can be approved or rejected before its evidence influences
the next version. v0.77 adds `training_recipe.json`, so the plan also links the
recipe that can reconstruct the run. v0.78 adds transformer responsibility
surfaces that consume the same training plan, recipe, verifier, and promotion
artifacts without changing their names. v0.79 adds model/config and checkpoint
metadata surfaces that keep run metadata aligned with those artifacts. v0.80
adds eval/checkpoint-load surfaces that preserve eval artifact shapes while
moving report mechanics out of the transformer monolith. v0.81 uses those
surfaces for a profile target-share objective repair while preserving the same
training-plan and replay-plan evidence trail. v0.82 screens that objective with
the same artifacts, rejects the result on branch diversity, and keeps the
corpus-hygiene trail as part of the evidence bundle rather than a quality
promotion shortcut. v0.83 adds prompt-specific ownership margins and runs the
same governed screen path; the artifacts still reject promotion when trained
snapshots lose target-token coverage. v0.84 adds baseline replay anchors and
keeps the same artifact trail, again rejecting promotion when trained snapshots
miss the baseline coverage floor. v0.85 adds baseline-floor update gating and
uses the same artifact trail to reject all unsafe attempted updates before they
can become trusted model state. v0.86 adds adaptive baseline-floor retries and
uses the same artifact trail to reject all `200/200` retry attempts before they
can become trusted model state. v0.87 adds baseline-covered repair retries and
uses the same artifact trail to reject all `200/200` repaired attempts before
they can become trusted model state.
