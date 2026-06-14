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
| `corpus_snapshot.json` | Current ledger source hashes and record counts. |
| `corpus_diff.json` | Comparison to the previous promoted run. |
| README / Docusaurus / marketing | Public state that must not drift. |

Start with [release discipline](./release-discipline.md), then read
[experiment registry](./experiment-registry.md), [corpus hygiene](./corpus-hygiene.md),
[provenance](./provenance.md), and [docs drift](./docs-drift.md).
