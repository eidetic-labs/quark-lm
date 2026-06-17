---
title: Historical Evidence Archive
description: Earlier QuarkLM run evidence moved out of GOAL.md.
---

# Historical Evidence Archive

This page preserves older run evidence that used to live in `GOAL.md`. It is a
record, not a status report. `GOAL.md` is now the durable goal contract; current
status, release posture, and the latest transformer evidence live in
[Current evidence](./current-evidence.mdx), `STATUS.md`, and
`sites/shared/current-state.json`.

The archive exists so that evidence is never lost when it leaves the
current-state surfaces. QuarkLM keeps failed and superseded runs as versioned
diagnostic evidence rather than discarding them, so the chain of what was tried,
what passed, and what was rejected stays inspectable. Moving an entry here is how
that history is retained without letting old numbers drift back into the goal
contract or the README.

## How to read this archive

The same distinction enforced everywhere in QuarkLM applies to every row below.
A run that answers a probe correctly proves the corpus *contains* the answer; it
does not prove the transformer *learned* it. Two tracks ran in parallel, and the
archive keeps them apart:

| Track | What its evidence means |
| --- | --- |
| Responder / learned-component track | The deterministic responder, retrieval memory, the learned answer model, and the answer decoder serve admitted knowledge. Exact results here are `memory-served` or learned-classifier evidence, not from-scratch transformer promotion. |
| Transformer screen track | The from-scratch decoder-only transformer is the `weight-consolidation` path. Its screens stayed separate from promoted responder evidence and are still blocked on `branch_diversity_target`. |

See [Language model](./language-model.md) for the three evidence states and
[Build](../build/index.md) for the two paths that produce them.

## Early learned components

These runs built and tightened the responder track's learned components before
the transformer architecture work began. They move classifier and decoder
exactness, not from-scratch transformer weights.

| Run | Archived signal |
| --- | --- |
| `runs/context64-v0.2/` | Character model validation NLL moved `3.4968 -> 2.6545`; known QA target NLL moved `3.4979 -> 2.4155`; held-out target NLL moved `3.4978 -> 2.5788`; free-form exact remained `0`. |
| `runs/answer-v0.1/` | Learned answer model moved QA, unknown, held-out, and paraphrase exactness from weak baselines to full exactness before stricter unseen-paraphrase tightening. |
| `runs/answer-v0.2/` | Learned answer model passed stricter unseen paraphrase probes: QA `8/8`, unknown `4/4`, held-out `8/8`, paraphrase `8/8`. |
| `runs/decoder-v0.2/` | Generative answer decoder moved from `0/8`, `0/4`, `0/8`, `0/8` exactness to QA `8/8`, unknown `4/4`, held-out `8/8`, paraphrase `8/8`. |

## Early self-improvement runs

These runs built the admission, audit, and self-diagnosis discipline that the
loop still depends on. They show the corpus growing through ledgered admissions
and the probes being generated from those admitted sources, so a passing probe is
evidence the corpus can answer it rather than a hand-written test.

| Run | Archived signal |
| --- | --- |
| `runs/self-improve-v0.9/` | Stricter lesson split kept held-out facts out of exact held-out prompt training; prompt leakage audit passed; answer model and decoder passed QA, unknown, held-out, and paraphrase evals. |
| `runs/self-improve-v0.12/` | Added operational self and learning-admission concepts plus the first admitted memory event; answer model and decoder passed owner, self, learning, and admissions evals. |
| `runs/self-improve-v0.14/` | Expanded admitted memory log to two facts; admission probes expanded to `8`; forgetting and prompt leakage audits passed. |
| `runs/self-improve-v0.16/` | Moved provenance code into `provenance`; wrote corpus snapshots and diffs; forgetting and leakage audits passed. |
| `runs/self-improve-v0.17/` | Generated admission probes from `corpus/admissions.jsonl`; probe sync passed with zero missing, extra, or mismatched ids. |
| `runs/self-improve-v0.18/` | Renamed the product to QuarkLM, added `quark-lm-*` script aliases, and generated admission paraphrase probes. |
| `runs/self-improve-v0.19/` | Added glossary word `stone` and admitted `learned-ivy-stone`; direct probes reached `12/12`, paraphrase probes `21/21`, and bridge lessons protected held-out transfer. |
| `runs/self-improve-v0.20/` | Generated glossary probes from `corpus/glossary.json`; glossary probes passed `20/20`; exact eval audit and promotion gate passed. |
| `runs/self-improve-v0.21/` | Added glossary words `shell`, `coin`, and `drum`; admitted three new memories; direct probes reached `24/24`, paraphrase probes `42/42`, glossary probes `26/26`; rule-based self-diagnosis reported `uses_external_model: false`. |
| `runs/self-improve-v0.22/` | Expanded operational self facts and learning rules, added self-diagnosis corpus facts, and exposed the need to preserve failed-attempt evidence. |
| `runs/self-improve-v0.23/` | Attempt archives became part of the loop so failed gates remain preserved instead of being overwritten by repair attempts. |
| `runs/self-improve-v0.24/` | First transformer architecture work was kept separate from promoted responder evidence. |
| `runs/self-improve-v0.25/` through `runs/self-improve-v0.42/` | Continued the promoted responder track while transformer screens stayed separate until neural promotion gates mature. Current promoted responder evidence remains `runs/self-improve-v0.42/`. |

## Transformer evidence index

The transformer run history is now documented primarily in
[Transformer](../build/transformer.md), [Provenance](../operate/provenance.md),
and [Current evidence](./current-evidence.mdx). This index keeps the major phases
the old `GOAL.md` evidence section recorded, as a map into that detail.

Read each phase against the same rule: candidate-selector and generator results
are auxiliary evidence, while raw greedy transformer answers are the only signal
that would count toward neural promotion. None of these phases cleared the
branch-diversity gate.

| Phase | Representative runs | Archived signal |
| --- | --- | --- |
| Architecture start | `runs/transformer-v0.24/`, `runs/transformer-v0.25/` | Tiny decoder-only transformer from random weights using the corpus-trained character tokenizer. |
| Answer training start | `runs/transformer-answer-v0.26/`, `runs/transformer-answer-v0.27/` | First transformer answer-training and faster eval-scoped candidate evaluator. |
| Choice/selector path | `runs/transformer-answer-v0.28-choice-prefix-pilot/`, `runs/transformer-answer-v0.29-selector-fast/`, `runs/transformer-answer-v0.30-selector-emission/` | Candidate-selector evidence improved answer selection while raw greedy generation stayed weak. |
| Generator path | `runs/transformer-answer-v0.31-generator-weighted-lr035-80k/` | No-candidate auxiliary generator moved exact generation from `0/219 -> 219/219`; this remains generator evidence, not transformer greedy promotion. |
| Direct-answer repair | `runs/transformer-answer-v0.32-direct-base-context32/` through `runs/transformer-answer-v0.42-branch-repair-contrast50-dim8-context32/` | Direct-answer modes improved distributional metrics and candidate behavior but did not make raw greedy transformer answers reliable. |
| Branch diagnostics | `runs/transformer-answer-v0.43-branch-profile-smoke-dim4-context16/` through `runs/transformer-answer-v0.43-branch-diversity-target-smoke-dim4-context80/` | Branch profiles, context coverage, and branch-diversity targets exposed prompt-independent first-token collapse. |
| Representation screens | `runs/transformer-answer-v0.43-context-mean-branch-batch-smoke-dim4-context16/` through `runs/transformer-answer-v0.43-prompt-position-scale32-repcontrast50-smoke-dim4-context80/` | Context summaries, projections, prompt attention, prompt-position projections, and representation contrast moved measured surfaces but did not pass branch diversity. |
| Structure audit and pre-layer norm | `STRUCTURE_AUDIT.md`, `runs/transformer-answer-v0.44-prelayernorm-repcontrast50-prompt-position-smoke-dim4-context80/`, `runs/transformer-answer-v0.44-target-balanced-prelayernorm-repcontrast50-prompt-position-smoke-dim4-context80/` | Open-source structure was studied as reference only; pre-layer-norm partially cracked non-QA collapse but remained rejected because formal branch-diversity gates failed. |

The generator reaching `219/219` while the direct transformer stayed at `0/219`
is the archived form of the distinction the project still holds today: the system
could already *serve* every answer while the neural weights had not yet *learned*
to route them. The complete version-by-version log continues in
[Transformer screen history](../build/transformer-screen-history.md).

## Archive rule

Historical evidence should not drift back into `GOAL.md` or the README.
Version-specific detail belongs on this page only when it is archival context.
Current release evidence belongs in [Current evidence](./current-evidence.mdx),
the shared current state, and the relevant Build or Operate docs.
