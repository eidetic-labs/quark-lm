---
title: Tokenizer Manifests
description: How QuarkLM records corpus-only tokenizer updates.
---

# Tokenizer Manifests

<p className="qlm-meta"><span>5 min read</span><span>For operators</span><span>Updated 2026-06-18</span></p>

<div className="qlm-lead">

**What you will learn**

- Which files prove a tokenizer was trained only from admitted text.
- How accepted and rejected candidate tokens are recorded.
- Which gates block tokenizer changes from becoming hidden model claims.

</div>

Tokenizer updates are evidence-bearing changes. A subword tokenizer can reduce
long-answer token count, but it can also hide memorization if whole answers are
accepted as tokens. QuarkLM records tokenizer updates with two artifacts:
`tokenizer_manifest.json` and `tokenizer_report.json`.

The self-improvement answer cycle now writes both artifacts for every attempt.
They are candidate evidence only: the active tokenizer remains unchanged until a
separate model-evaluation screen proves the candidate improves behavior without
regressing retention, unknown policy, leakage, or branch diversity.

## Manifest

`tokenizer_manifest.json` records the vocabulary proposal.

| Field | Meaning |
| --- | --- |
| `tokenizer_type` | The tokenizer implementation used by the run. |
| `corpus_hash` | Hash of the admitted text used to score candidates. |
| `source_files` | Corpus files used as the tokenizer source. |
| `tokens` / `merge_rules` | Append-only token list and replayable merge rules. |
| `candidate_scores` | Accepted candidates with frequency, context diversity, and score. |
| `rejected_candidates` | Candidates rejected by guards such as full-answer or excessive-length checks. |
| `purity` | Explicit false flags for pretrained tokenizer and external vocabulary use. |

## Report

`tokenizer_report.json` records the behavioral summary:

- exact round-trip status;
- character token count versus subword token count;
- compression ratio and token-count savings;
- full-answer-token audit;
- average context-diversity score;
- tokenizer-level long-answer savings;
- whether model-level long-answer effect still needs transformer diagnostics.

The report is not a neural promotion artifact. It is a tokenizer artifact that
can become an input to a transformer screen.

Before self-improvement writes a tokenizer candidate, the manifest/report pair
must pass standalone validation. The validator checks schema version, stable
manifest hash, corpus hash shape, purity flags, exact round-trip, report math,
and protected full-answer-token rejection.

## Guard evidence

Self-improvement reports include `tokenizer_candidate_guard`, a compact
promotion check that must pass before the answer-cycle result can promote. It
requires:

- a recorded tokenizer candidate;
- no silent active-tokenizer promotion;
- exact round-trip;
- zero protected full-answer tokens;
- `pretrained_tokenizer: false`;
- `external_vocabulary: false`;
- `admitted_corpus_only: true`.

This guard keeps tokenizer progress visible without letting compression evidence
be mistaken for learned-model evidence.

## Promotion rules

A tokenizer update is rejected if it:

- remaps existing token IDs;
- imports pretrained vocabulary or normalization;
- fails exact round-trip;
- admits a full answer as a token;
- improves long-answer token count while regressing short-answer or retention evidence;
- weakens unknown-policy, leakage, or branch-diversity gates in the model screen.

## Operating pattern

Run tokenizer experiments as small screens first. Compare the character
baseline against the subword tokenizer with the same corpus, model size, and
epoch budget. Promote only the evidence state that actually passed:
compression, manifest purity, benchmark diagnostics, or neural consolidation.

## What is next

<div className="qlm-next">

<a href="../../build/tokenizer/"><strong>Build</strong><span>Tokenizer</span><small>How the closed-world subword tokenizer works.</small></a>

<a href="./training-recipes/"><strong>Operate</strong><span>Training recipes</span><small>Where tokenizer identity belongs in reproducible runs.</small></a>

<a href="../../secure/purity-boundary/"><strong>Secure</strong><span>Purity boundary</span><small>The boundary that keeps tokenizer growth auditable.</small></a>

</div>
