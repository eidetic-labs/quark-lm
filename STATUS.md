# QuarkLM - Status

**Status:** Experimental research scaffold
**Active version:** v0.42 wider sparse branch-contrast transformer training
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

`runs/self-improve-v0.42/` passes protected prompt leakage, forgetting against
`runs/self-improve-v0.41/`, exact eval audit, promotion gate, and reaches 100%
exact match for the responder, learned answer classifier, and generative answer
decoder across all 10 current eval sets. Admission probes now pass `48/48`
direct and `84/84` paraphrase records; glossary probes pass `38/38`. The
passing attempt is archived at
`runs/self-improve-v0.42/attempts/attempt-001/`. The report diagnosis records
zero blockers with `uses_external_model: false`.

`runs/transformer-answer-v0.42-branch-repair-contrast50-dim8-context32/` is the current
from-scratch transformer answer evidence. It uses the corpus-trained character
tokenizer, no pretrained weights, no pretrained tokenizer, and no external
embeddings. v0.42 keeps the v0.41 sparse branch-repair/contrast objective and
widens the from-scratch transformer from embedding/feed-forward dimensions
`4/8` to `8/16`. The run trained `80` target-loss steps plus `1000` sparse
branch repair/contrast direct answer steps with context size `32`; answer
target NLL moved `3.5850 -> 2.4129`, direct answer target loss moved
`3.4278 -> 2.2708`, and transformer-only eval-scoped candidate accuracy moved
`15/219 -> 37/219`. Raw direct greedy exact remained `0/219 -> 0/219`; the
failure changed from a repeated `"te"`/`"e"` loop to the short wrong answer
`" te."`, so prompt-conditioned greedy branching is still the current
bottleneck.

Unpromoted v0.43 work added three pieces of transformer-loop evidence without
changing the promoted checkpoint. The forward pass now computes only the final
position consumed by the language-model head, cutting the transformer unit-test
runtime from roughly `13.9s` to `6.2s` on this machine. Transformer answer runs
now record prompt context-coverage metrics, showing that context size `80`
covers all current semantic eval templates (`219/219`) while context size `32`
does not. The hard-negative branch-contrast pilot at
`runs/transformer-answer-v0.43-hard-branch-contrast4-dim8-context32/` preserved
`37/219` candidates but regressed direct loss to `2.4225`, answer NLL to
`2.5402`, and greedy output to a repeated `" a"` loop. The full-context pilot at
`runs/transformer-answer-v0.43-branch-repair-contrast50-dim8-context80/`
preserved `37/219` candidates with `219/219` coverage but still trailed v0.42
on direct loss (`2.3122`) and answer NLL (`2.4546`). A 1500-step context-80 run
reached `38/219` candidates but regressed loss, NLL, and greedy output, so it
was not promoted. A layer-normalized context-80 screen at
`runs/transformer-answer-v0.43-layernorm-screen-dim8-context80/` preserved full
coverage and `37/219` candidates but regressed answer NLL to `2.5881` and
collapsed greedy output into repeated `" y"`/`"e"` loops, so it also remains
unpromoted evidence. A branch-span screen at
`runs/transformer-answer-v0.43-branch-span3-screen-dim8-context32/` broadened
branch repair to answer positions `1..3`; it preserved `37/219` candidates but
regressed answer NLL to `2.7426` and produced a long `"neeee"` loop, so it was
not promoted. Multi-layer transformer support was added as an explicit
architecture option, but the first two-layer context-32 screen at
`runs/transformer-answer-v0.43-two-layer-screen-dim8-context32/` was interrupted
before final direct-answer metrics because the full-block scalar autograd path
was too slow for the regular loop. The partial JSONL history remains runtime
evidence only. A follow-up optimized the final layer of stacked transformers to
compute only the final state and added equivalence coverage against full-stack
logits, but
`runs/transformer-answer-v0.43-two-layer-finalopt-screen-dim8-context32/` still
interrupted before final metrics because the intermediate full-state layer and
positive-context repair update remain too expensive. A follow-up added
top-layer-only direct-answer updates for stacked transformers and the explicit
`--skip-post-direct-snapshot` screening flag. The completed bounded screen at
`runs/transformer-answer-v0.43-two-layer-toponly-skip-screen-dim8-context32/`
saved a two-layer checkpoint after `40` target-loss steps and `80` top-layer
direct steps, recorded that the post-direct candidate snapshot was skipped,
improved direct-answer target loss `3.5186 -> 3.2436`, but kept direct greedy
exact at `0/219 -> 0/219` with repeated `"a"` output. It is runtime and
training-loop evidence only; v0.42 remains the promoted transformer checkpoint.
Direct-answer snapshots now include branch profiles computed from QuarkLM's own
logits. The smoke run at
`runs/transformer-answer-v0.43-branch-profile-smoke-dim4-context16/` recorded
QA branch-position-1 accuracy at `1/8` before and after five tiny direct updates;
the dominant prediction moved from all `"o"` to all `"y"`, and the average
target margin stayed negative. This gives the next repair loop a measurable
prompt-independent branch-collapse signal. Branch-collapse repair now uses the
dominant sampled branch prediction as the unlikelihood negative. The full-dose
smoke at `runs/transformer-answer-v0.43-branch-collapse-smoke-dim4-context16/`
regressed direct loss and moved the QA branch collapse to all `"a"` predictions.
The periodic smoke at
`runs/transformer-answer-v0.43-periodic-branch-collapse-smoke-dim4-context16/`
improved direct loss `3.5800 -> 3.5157`, but QA branch accuracy stayed
`1/8 -> 1/8` and the dominant branch prediction moved from all `"o"` to all
`"n"`. The repair is recorded as rejected evidence: dominant-token suppression
helps loss under sparse dosage, but does not create prompt-specific branches.
Branch-batch contrast now trains multiple distinct branch targets in a single
update. The full-dose smoke at
`runs/transformer-answer-v0.43-branch-batch-smoke-dim4-context16/` improved loss
only slightly and moved the QA branch collapse to all `"y"` predictions. The
periodic smoke at
`runs/transformer-answer-v0.43-periodic-branch-batch-smoke-dim4-context16/`
improved direct loss `3.5800 -> 3.5248`, but QA branch accuracy regressed
`1/8 -> 0/8` and the dominant branch prediction moved to all `"a"`. That makes
branch-batch contrast rejected repair evidence too: the objective can move loss
without forcing prompt-conditioned branch separation.

A representation-side screen added `--use-context-mean`, which adds the
mean-pooled prompt context to the final transformer hidden state. The
branch-batch comparison run at
`runs/transformer-answer-v0.43-context-mean-branch-batch-smoke-dim4-context16/`
improved direct loss `3.5805 -> 3.5252`; the selected branch-repair comparison
at
`runs/transformer-answer-v0.43-context-mean-branch-repair-smoke-dim4-context16/`
improved direct loss `3.5805 -> 3.5310`. Both regressed QA branch accuracy
`1/8 -> 0/8` and collapsed to all `"a"` predictions, so context averaging is
rejected representation evidence rather than a promoted transformer step.

A follow-up representation screen added `--use-context-projection`, a
zero-initialized trainable projection of the mean-pooled context. The
branch-repair run at
`runs/transformer-answer-v0.43-context-projection-branch-repair-smoke-dim4-context16/`
moved all `20` projection parameters, improved direct loss
`3.5802 -> 3.5217`, and the branch-batch comparison at
`runs/transformer-answer-v0.43-context-projection-branch-batch-smoke-dim4-context16/`
improved direct loss `3.5802 -> 3.5252`. Both regressed QA branch accuracy
`1/8 -> 0/8` and collapsed to all `"a"` predictions, so learned prompt-summary
projection is rejected representation evidence too.

A stronger representation screen added `--use-prompt-attention-summary`, a
trainable attention-pooled context summary with a zero-initialized output
projection. The branch-repair run at
`runs/transformer-answer-v0.43-prompt-attention-branch-repair-smoke-dim4-context16/`
moved all `20` zero-initialized output projection parameters and improved
direct loss `3.5802 -> 3.5217`; the branch-batch comparison at
`runs/transformer-answer-v0.43-prompt-attention-branch-batch-smoke-dim4-context16/`
improved direct loss `3.5802 -> 3.5252`. Both regressed QA branch accuracy
`1/8 -> 0/8` and collapsed to all `"a"` predictions, so prompt attention is
also rejected representation evidence.

Direct-answer snapshots now include branch-context coverage diagnostics: the
visible branch context text, semantic feature coverage, context collisions, and
target-token ambiguity. The context-16 screen at
`runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context16/`
showed QA branch contexts had `0/8` semantic coverage and `4` ambiguous branch
windows. The context-32 screen at
`runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context32/`
removed QA ambiguity but still had `0/8` semantic coverage. The tiny
context-80 screen at
`runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context80/`
reached complete branch-context coverage across all eval sets (`219/219`) with
zero ambiguous branch contexts. This makes efficient longer-context branch
repair the next structured transformer target.

The branch-context coverage diagnostic is now actionable through
`--direct-answer-require-branch-context-gate`. The context-16 gated screen at
`runs/transformer-answer-v0.43-branch-context-gate-smoke-dim4-context16/`
required the gate, failed it, and recorded `actual_steps: 0` for `5` requested
direct steps. The context-80 gated screen at
`runs/transformer-answer-v0.43-branch-context-gate-smoke-dim4-context80/`
passed the gate and recorded `actual_steps: 1` for `1` requested direct step.
This is a training-loop guardrail, not promoted model quality evidence.

Direct-answer snapshots now have a `branch-only` mode for bounded longer-context
screens. The mode skips greedy completion evals in JSONL snapshots while still
recording branch profiles, branch-context coverage, and the gate result. The
context-80 gated screen at
`runs/transformer-answer-v0.43-branch-context-gated-branchonly-smoke-dim4-context80/`
passed the required gate across all `219/219` semantic records, recorded
`actual_steps: 5` for `5` requested direct branch-repair steps, and marked
`direct_answer_evals_skipped: true`. This makes longer-context branch-repair
screening cheaper and auditable, but it is not promoted quality evidence because
greedy completion evals were intentionally skipped.

Two dim8 context-80 branch-only follow-up screens used that cheaper evidence
path to test whether the best prior sparse repair/contrast policy or
branch-batch contrast could create prompt-specific branches once the visible
context was complete. The periodic repair/contrast screen at
`runs/transformer-answer-v0.43-branchonly-periodic-repair-contrast50-dim8-context80/`
passed the required gate, ran `100/100` direct steps, and lowered interval train
loss `6.7890 -> 6.4326`, but final QA branch prediction collapsed from all
space to all `"a"` with final QA branch accuracy `0/8`. The branch-batch screen
at `runs/transformer-answer-v0.43-branchonly-branch-batch-dim8-context80/`
passed the same gate, ran `50/50` direct steps, and lowered interval train loss
`3.4614 -> 3.1976`, but QA branch prediction still collapsed to all `"a"` with
final QA branch accuracy `0/8`. These are rejected screening results: complete
context and lower branch loss are still insufficient without a prompt-specific
branch signal.

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
