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
- replay-plan path and summary when profile-aware replay writes one;
- planned artifacts.

Candidate ratio is reported in v0.73. Candidate quarantine is the next planned
mechanic, so generated or proposed examples still must not become training data
without a later admission and verification path.
