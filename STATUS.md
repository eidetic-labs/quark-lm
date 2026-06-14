# QuarkLM - Status

**Status:** Experimental research scaffold
**Active version:** v0.41 sparse branch-contrast direct transformer training
**Last updated:** 2026-06-14
**Buildable:** yes, with Python standard library only

QuarkLM explores bounded epistemic growth: a model starts from random weights
and only learns from an explicitly admitted corpus. The intended GitHub
repository slug is `quark-lm`. The first milestone is a dependency-free toy
learner, not a production model.

Working tagline: Big idea. Tiny package.

## Current Scope

- Human-authored seed glossary, grammar, and admitted-memory log.
- Deterministic curriculum generation.
- Character tokenizer trained only on generated admitted text.
- Tiny neural character MLP trained from random initialization.
- Tiny decoder-only transformer trained from random initialization with a
  dependency-free scalar autodiff engine.
- Reliable corpus responder for exact closed-world answers.
- Learned answer classifier trained from random weights.
- Generative answer decoder trained from random weights.
- Operational self facts: dataset boundary, pretrained-weight policy, unknown
  policy, and improvement method.
- Admission rule: "I learned something new" means a fact was admitted into the
  ledgered corpus before training and weight updates.
- Batch admission support for appending multiple structured memories with
  duplicate-id rejection before writing.
- Forgetting audit support for comparing a new self-improvement cycle against a
  prior report.
- Corpus provenance snapshots and corpus diffs for self-improvement reports.
- Generated direct and paraphrase admission probes from
  `corpus/admissions.jsonl`, with probe-sync audits in self-improvement
  reports.
- Generated glossary probes from `corpus/glossary.json` probe words, with
  glossary-probe audits in self-improvement reports.
- Admitted memories and story facts now produce generated bridge lessons to
  preserve held-out transfer without leaking protected held-out prompts.
- Learned-component eval summaries include failed records when any probe misses.
- Self-improvement reports include an exact eval audit and a promotion gate; the
  command returns failure unless all audits and evals are promotion-ready.
- Self-improvement reports include rule-based self-diagnosis that recommends
  the next action from report evidence without using an external model.
- Self-improvement attempts are archived under `attempts/attempt-###/` before
  the top-level latest report is updated.
- Package metadata now uses `quark-lm`, with `quark-lm-*` script aliases.
- Public surfaces: Docusaurus docs at `docs.quark-lm.eidetic-labs.com` and a
  standalone static marketing page at `quark-lm.eidetic-labs.com`, with
  GitHub Actions deployment scaffolds. The marketing site is not Docusaurus.
- SOLID-aligned quality guidance in `QUALITY.md`.
- Source probes for known, unknown, held-out, paraphrase, ownership, self,
  learning, admission, admission-paraphrase, and glossary answers.

## Latest Evidence

`runs/self-improve-v0.41/` passes protected prompt leakage, forgetting against
`runs/self-improve-v0.40/`, exact eval audit, promotion gate, and reaches 100%
exact match for the responder, learned answer classifier, and generative answer
decoder across all 10 current eval sets. Admission probes now pass `48/48`
direct and `84/84` paraphrase records; glossary probes pass `38/38`. The
passing attempt is archived at
`runs/self-improve-v0.41/attempts/attempt-001/`. The report diagnosis records
zero blockers with `uses_external_model: false`.

`runs/transformer-answer-v0.41-branch-repair-contrast50-context32/` is the current
from-scratch transformer answer evidence. It uses the corpus-trained character
tokenizer, no pretrained weights, no pretrained tokenizer, and no external
embeddings. v0.41 keeps branch-repair training at answer content position `1`
as the base update and injects prompt-contrast branch training every `50` direct
steps. The run trained `80` target-loss steps plus `1000` sparse branch
repair/contrast direct answer steps with context size `32`; answer target NLL
moved `3.5828 -> 2.4734`, direct answer target loss moved `3.3496 -> 2.3315`,
and transformer-only eval-scoped candidate accuracy moved `15/219 -> 37/219`.
Raw direct greedy exact remained `0/219 -> 0/219`; the failure changed from a
repeated `"ten"` loop to a repeated `"te"`/`"e"` loop, so prompt-conditioned
greedy branching is still the current bottleneck.

The v0.31 no-candidate auxiliary generator remains the best no-candidate exact
answer evidence: `runs/transformer-answer-v0.31-generator-weighted-lr035-80k/`
trained the generator for `80000` weighted steps at learning rate `0.035` and
moved exact generation from `0/219 -> 219/219` with
`uses_answer_candidates: false`.

## Out Of Scope

- Pretrained foundation models.
- Pretrained tokenizers or embedding models.
- Web retrieval.
- Production-grade model quality.
