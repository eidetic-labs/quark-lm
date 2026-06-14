# QuarkLM

QuarkLM is a tiny research prototype for a model that only learns from an
explicitly admitted corpus. The intended GitHub repository slug is
`quark-lm`. The current Python import path remains `closed_world_lm` until a
dedicated package migration is promoted.

Tagline: Big idea. Tiny package.

The v0 experiment is intentionally modest:

- no pretrained weights
- no pretrained tokenizer
- no external embeddings
- no runtime dependencies outside the Python standard library for the model
  prototype
- character-level vocabulary learned from the admitted corpus
- a tiny neural character model initialized from random weights
- a tiny decoder-only transformer initialized from random weights

This is not yet a useful assistant. It is a first closed-world language model loop:
glossary, simple grammar, tiny stories, question-answer lessons, training, and
closed-world probes.

## Latest Evidence

Current promoted run: `runs/self-improve-v0.42/`.
Current transformer answer-lesson run:
`runs/transformer-answer-v0.42-branch-repair-contrast50-dim8-context32/`.

- v0.42 keeps the admitted corpus unchanged from v0.41 at `12` admitted facts.
- The v0.42 self-improvement cycle passed on archived `attempt-001`, with
  forgetting compared against `runs/self-improve-v0.41/`.
- Admission-probe audit passed: direct probes `48/48`, paraphrase probes
  `84/84`, no missing, extra, or mismatched ids.
- Glossary-probe audit passed: glossary probes `38/38`, no missing, extra, or
  mismatched ids.
- v0.42 keeps sparse prompt-contrast branch repair and widens the from-scratch
  transformer from embedding/feed-forward dimensions `4/8` to `8/16`. This is
  still random initialization with the corpus-trained character tokenizer.
- The current v0.42 direct-transformer run trained `80` target-loss steps plus
  `1000` sparse branch-repair/contrast direct answer steps at context size
  `32`. Direct answer target loss moved `3.4278 -> 2.2708`, transformer answer
  target NLL moved `3.5850 -> 2.4129`, and eval-scoped transformer-only
  candidate accuracy moved `15/219 -> 37/219`.
- Raw direct greedy transformer exact remained `0/219 -> 0/219`; completions
  moved from the repeated `"te"`/`"e"` loop to the short wrong answer `" te."`.
  v0.42 improves the scored target distribution without damaging candidate
  discrimination and reduces runaway looping, but still shows that raw greedy
  answer emission needs stronger prompt-conditioned representation.
- Unpromoted v0.43 experiments added a faster final-position transformer
  forward pass, explicit prompt context-coverage metrics, and an experimental
  hard-negative branch-contrast mode. The hard-negative context-32 run
  (`runs/transformer-answer-v0.43-hard-branch-contrast4-dim8-context32/`) kept
  candidate accuracy at `37/219` but regressed direct loss to `2.4225`, answer
  NLL to `2.5402`, and collapsed greedy output to a repeated `" a"` loop. The
  context-80 run
  (`runs/transformer-answer-v0.43-branch-repair-contrast50-dim8-context80/`)
  achieved full semantic template coverage (`219/219`) and the short `" t."`
  failure, but still trailed v0.42 with direct loss `2.3122` and answer NLL
  `2.4546`. A longer context-80 1500-step run reached `38/219` candidates but
  regressed loss/NLL and greedy output, so v0.42 remains the promoted
  transformer checkpoint. A layer-normalized context-80 screen
  (`runs/transformer-answer-v0.43-layernorm-screen-dim8-context80/`) also kept
  `37/219` candidates with full context coverage, but regressed answer NLL to
  `2.5881` and produced repeated `" y"`/`"e"` loops, so it was not promoted.
  A branch-span screen
  (`runs/transformer-answer-v0.43-branch-span3-screen-dim8-context32/`) tested
  repairing answer positions `1..3`; it preserved `37/219` candidates but
  regressed answer NLL to `2.7426` and produced a long `"neeee"` loop, so it
  also remains unpromoted evidence.
  Multi-layer transformer support was added after that, but the first
  two-layer context-32 screen
  (`runs/transformer-answer-v0.43-two-layer-screen-dim8-context32/`) was
  interrupted before final direct-answer metrics because the full-block scalar
  autograd path was too slow for the regular loop. It produced only partial
  JSONL history and is runtime evidence, not promotion evidence. A follow-up
  optimized the final transformer layer to compute only the last state and
  proved equivalence against full-stack logits, but
  `runs/transformer-answer-v0.43-two-layer-finalopt-screen-dim8-context32/` was
  still interrupted before final metrics; the intermediate full-state layer is
  still too expensive for direct-answer repair updates. A follow-up added
  top-layer-only direct-answer training for stacked models plus an explicit
  `--skip-post-direct-snapshot` screening control. The completed bounded screen
  at
  `runs/transformer-answer-v0.43-two-layer-toponly-skip-screen-dim8-context32/`
  saved a two-layer checkpoint after `40` target-loss steps and `80` top-layer
  direct steps, recorded that the post-direct candidate snapshot was skipped,
  moved direct-answer target loss `3.5186 -> 3.2436`, and still failed direct
  greedy exact at `0/219 -> 0/219` with repeated `"a"` output. This is loop
  completion and runtime evidence, not promotion evidence. Direct-answer
  snapshots now also record branch profiles from QuarkLM's own logits. The
  smoke run at
  `runs/transformer-answer-v0.43-branch-profile-smoke-dim4-context16/` shows
  the QA branch-position-1 prediction collapsed from all `"o"` choices at
  baseline to all `"y"` choices after five direct updates, with branch accuracy
  `1/8` and a negative average target margin. That is diagnostic evidence for
  prompt-independent branch collapse, not a promotion candidate. Branch-collapse
  repair then used the dominant branch token as the unlikelihood negative. The
  full-dose smoke
  `runs/transformer-answer-v0.43-branch-collapse-smoke-dim4-context16/`
  regressed loss and moved collapse to all `"a"` predictions; the periodic
  smoke
  `runs/transformer-answer-v0.43-periodic-branch-collapse-smoke-dim4-context16/`
  improved direct loss `3.5800 -> 3.5157` but still stayed at QA branch
  accuracy `1/8` and collapsed to all `"n"` predictions. The lesson is that
  dominant-token suppression alone is not enough to create prompt-specific
  branches. Branch-batch contrast then trained several distinct branch targets
  in one update. The full-dose smoke
  `runs/transformer-answer-v0.43-branch-batch-smoke-dim4-context16/` improved
  loss only slightly and collapsed to all `"y"` predictions; the periodic smoke
  `runs/transformer-answer-v0.43-periodic-branch-batch-smoke-dim4-context16/`
  improved direct loss `3.5800 -> 3.5248` but regressed QA branch accuracy
  `1/8 -> 0/8` and collapsed to all `"a"` predictions. The current evidence
  points at weak prompt representation rather than insufficient branch-token
  suppression.
- The v0.31 no-candidate auxiliary generator remains the best exact
  no-candidate answer evidence: it trained for `80000` weighted steps at
  learning rate `0.035` and moved exact generation from `0/219 -> 219/219` with
  `uses_answer_candidates: false`.
- The transformer uses QuarkLM's existing corpus-trained character tokenizer,
  learned token and position embeddings, one causal self-attention block, a
  feed-forward block, and a next-character language-model head.
- v0.23 introduced attempt archives. A failed gate remains preserved at
  `runs/self-improve-v0.23/attempts/attempt-001/`, while the repaired passing
  attempt remains preserved at `runs/self-improve-v0.23/attempts/attempt-002/`.
- Direct and paraphrase admission probes are generated from
  `corpus/admissions.jsonl` and audited in the run report.
- Self and learning evals expanded to `7/7` and `4/4`, including questions about
  self-diagnosis source, external model shaping, and repair-action selection.
- Forgetting audit passed against `runs/self-improve-v0.41/`.
- Protected prompt leakage audit passed.
- Learned eval summaries now retain `failed_records` so failed cycles are
  diagnosable from report artifacts.
- Self-improvement runs now include an exact eval audit and a promotion gate;
  the command exits non-zero unless generated probes, prompt leakage, forgetting,
  and exact eval audits all pass.
- Self-improvement reports now include a rule-based `self_diagnosis` section
  that uses no external model and recommends the next action from the report.
- Responder, learned answer classifier, and generative answer decoder reached
  100% exact match across QA, unknowns, held-out, paraphrases, ownership, self,
  learning, admissions, admission-paraphrase, and glossary evals.

## Quick Start

Run from this directory:

```bash
PYTHONPATH=src python3 -m closed_world_lm.curriculum --output build
PYTHONPATH=src python3 -m closed_world_lm.train --steps 300 --run runs/smoke
PYTHONPATH=src python3 -m closed_world_lm.evaluate --checkpoint runs/smoke/checkpoint.json
PYTHONPATH=src python3 -m closed_world_lm.respond --eval --json runs/smoke/respond.json
PYTHONPATH=src python3 -m closed_world_lm.answer_model train --run runs/answer-smoke
PYTHONPATH=src python3 -m closed_world_lm.answer_model eval \
  --checkpoint runs/answer-smoke/answer_model.json \
  --json runs/answer-smoke/answer_eval.json
PYTHONPATH=src python3 -m closed_world_lm.answer_decoder train --run runs/decoder-smoke
PYTHONPATH=src python3 -m closed_world_lm.answer_decoder eval \
  --checkpoint runs/decoder-smoke/answer_decoder.json \
  --json runs/decoder-smoke/decoder_eval.json
PYTHONPATH=src python3 -m closed_world_lm.self_improve answer-cycle \
  --run runs/self-improve-smoke
PYTHONPATH=src python3 -m closed_world_lm.transformer_char_model train \
  --run runs/transformer-smoke \
  --steps 20 \
  --context-size 8
PYTHONPATH=src python3 -m closed_world_lm.transformer_char_model answer-train \
  --run runs/transformer-answer-smoke \
  --steps 100 \
  --eval-every 0 \
  --candidate-scope eval \
  --selector-steps 200 \
  --selector-eval-every 0 \
  --selector-emit-completions \
  --generator-steps 400 \
  --generator-eval-every 0 \
  --direct-answer-steps 100 \
  --direct-answer-eval-every 0 \
  --direct-answer-mode periodic-branch-repair-contrast-unlikelihood \
  --direct-answer-negative-weight 1.0 \
  --direct-answer-positive-weight 1.0 \
  --direct-answer-contrast-weight 1.0 \
  --direct-answer-branch-position 1 \
  --direct-answer-rollout-interval 50 \
  --embedding-dim 8 \
  --feedforward-dim 16
```

The `closed_world_lm` module name is still the stable command path during this
phase. Installed console scripts also include `quark-lm-*` aliases while the
older `closed-world-*` aliases remain available.

The short training run is a smoke test, not a quality target. Increase
`--steps` once you want to watch the toy model absorb the nursery corpus.
Each training run writes `metrics.jsonl`, including step `0` as the untrained
baseline and periodic probe losses for known, unknown, and held-out questions.
Evaluation reports both free-form exact match and closed-world candidate
accuracy. Candidate accuracy asks whether the model assigns the lowest loss to
the correct admitted answer among known corpus answers.

`closed_world_lm.respond` is a reliable corpus-only response model. It parses
the admitted `fact:` lines from `build/train.txt` and answers known, unknown,
and held-out probes exactly. This gives the project a grounded teacher while the
neural character model improves.

`closed_world_lm.answer_model` is a learned closed-world response model. It uses
randomly initialized softmax weights and trains on question/answer lessons
derived from admitted corpus facts. It records step-0 baseline metrics,
generated lessons, a checkpoint, and evaluation reports.

`closed_world_lm.answer_decoder` is a generative learned response model. It
updates random weights to emit answer characters one at a time from the prompt
and prior answer prefix, using only corpus-derived lessons. Its trainer uses a
weighted training pool so longer self-improvement lessons get enough updates
without changing the admitted lesson corpus.

`closed_world_lm.transformer_char_model` is an experimental decoder-only
transformer language model. It is built in the Python standard library with a
tiny scalar autodiff engine, starts from random weights, uses the corpus-trained
character tokenizer, and writes its own checkpoint and metrics. The
`answer-train` subcommand trains on corpus-derived `AnswerExample` lessons. It
can also train a closed-world answer candidate selector with random weights
using `--selector-steps`. Add `--selector-emit-completions` to record the
selector's chosen candidate as the emitted completion for exact-match evidence.
Add `--generator-steps` to train the transformer-guided character answer
generator without answer candidates. Selector-assisted emission,
generator-emitted answers, transformer-only NLL, and raw transformer free-form
generation stay separate in metrics. Add `--direct-answer-steps` to continue
updating the transformer weights for strict greedy answer completion with an
admitted terminator token; `--direct-answer-mode first-error` targets the first
current greedy mismatch, `first-error-unlikelihood` also penalizes the wrong
self-predicted token, `rollout-unlikelihood` trains on generated-prefix
contexts, `hybrid-unlikelihood` alternates first-error and rollout updates,
`staged-unlikelihood` runs them in phases, and `random-char` samples answer
characters. `periodic-rollout-unlikelihood` keeps most updates on first-error
repair while injecting rollout repair every N steps. `early-stop-unlikelihood`
targets premature terminator predictions, and
`periodic-early-stop-unlikelihood` injects that repair every N steps.
`repeat-loop-unlikelihood` targets repeated suffix patterns in the model's own
greedy rollout, and `periodic-repeat-loop-unlikelihood` injects that repair
every N steps. `balanced-repair-unlikelihood` pairs a self-generated repair
with a teacher-forced admitted continuation, and
`periodic-balanced-repair-unlikelihood` injects that balanced repair every N
steps. `generated-prefix-recovery-unlikelihood` trains after the first bad
generated prefix, while `sequence-repair-unlikelihood` samples greedy mistakes
across the correct admitted target prefix. The `periodic-*` sequence and
generated-prefix modes inject those repairs every N steps.
`loop-escape-unlikelihood` pairs repeated-loop penalties with admitted
continuations, and `branch-repair-unlikelihood` targets an answer branch
position such as the first content character after the leading answer space.
`branch-contrast-unlikelihood` contrasts that branch against another admitted
prompt with a different branch target; `periodic-branch-repair-contrast-*`
keeps branch repair as the base update and injects contrast every N steps.
The direct transformer path is not yet part of the promotion gate for reliable
answers.

`closed_world_lm.admit` appends a new structured memory to
`corpus/admissions.jsonl`. That is the operational meaning of "I learned
something new": the fact is admitted first, then `self_improve answer-cycle`
regenerates lessons and updates versioned weights. It can admit one fact from
CLI fields or a JSONL batch. When writing to the default project admissions
file, it also syncs generated admission probes to `evals/admissions.jsonl`.
It syncs admission paraphrase probes to `evals/admission_paraphrases.jsonl` at
the same time.

`closed_world_lm.admission_probes` regenerates or checks direct and paraphrase
admission probes from `corpus/admissions.jsonl`. This keeps admitted-memory
eval sets derived from the admitted-memory source instead of hand-maintained
beside it.

`closed_world_lm.glossary_probes` regenerates or checks glossary definition
probes from the `probe_words` listed in `corpus/glossary.json`. Glossary probes
are evaluation-only ledger sources; glossary definitions become trainable
through the admitted glossary itself.

`closed_world_lm.self_improve answer-cycle` runs a repeatable improvement cycle:
regenerate the admitted curriculum, train the learned answer model, evaluate the
reliable responder, learned answer model, and generative answer decoder, then
write `self_improvement_report.json`. The report records the admission source,
direct/paraphrase admission-probe audit, glossary-probe audit, weight updates,
prompt-leakage audit, exact eval audit, promotion gate, rule-based
self-diagnosis, and exact evals for QA, unknowns, held-out, paraphrases,
ownership, self, learning, admissions, admission paraphrases, and glossary
definitions. Learned-component summaries include failed record details when an
eval is not exact.
Pass `--compare-report runs/<prior>/self_improvement_report.json` to add a
forgetting audit against a prior cycle. Each run archives every attempt under
`runs/<name>/attempts/attempt-###/`, then updates the top-level report as the
latest pointer. Each run also writes `corpus_snapshot.json` and
`corpus_diff.json` so source-level corpus changes can be audited alongside
weights and evals.

`closed_world_lm.self_diagnose` reads a `self_improvement_report.json` and emits
a deterministic repair recommendation from report evidence. It is intentionally
rule-based and sets `uses_external_model` to `false`; this is the current
non-external-model bridge toward future self-improvement guidance learned from
QuarkLM's own admitted artifacts.

## Admitting New Knowledge

To teach the model a new closed-world fact, admit it first:

```bash
PYTHONPATH=src python3 -m closed_world_lm.admit \
  --id learned-child-book \
  --person child \
  --object book \
  --color blue \
  --relation on \
  --container table
```

Use a fresh `--id` for each new admission. Duplicate ids are rejected. After
admission, run `closed_world_lm.self_improve answer-cycle` so the curriculum is
regenerated and the learned weights are updated from the new admitted data.
When using the default project paths, admission probes are regenerated
automatically. Use `--no-sync-probes` only when deliberately staging corpus and
eval changes separately.

To admit a batch, create a JSONL file with one object per fact using the same
fields, then run:

```bash
PYTHONPATH=src python3 -m closed_world_lm.admit \
  --batch path/to/new_admissions.jsonl
```

Batches are checked before writing: duplicate ids already in the corpus, or
duplicate ids inside the batch, reject the whole batch.

To check probe sync without writing:

```bash
PYTHONPATH=src python3 -m closed_world_lm.admission_probes --check
```

## Forgetting Checks

When adding new admissions, compare the new run to the last promoted report:

```bash
PYTHONPATH=src python3 -m closed_world_lm.self_improve answer-cycle \
  --run runs/self-improve-next \
  --compare-report runs/self-improve-v0.42/self_improvement_report.json
```

The forgetting audit compares responder, answer-classifier, and answer-decoder
final evals. For every shared eval set, the new run must keep at least the prior
probe count, exact matches, and exact rate.

## Engineering Quality

Project quality principles live in `QUALITY.md`. In short: keep module
responsibilities narrow, prefer pure functions for audits and corpus transforms,
add focused tests for behavior changes, and treat provenance artifacts as part
of the promotion gate. README, STATUS, GOAL, QUALITY, Docusaurus docs, and the
standalone marketing page should be updated with every promoted version so docs
do not drift from the current state. If they reference current product state,
release evidence, evals, or commands, they must be updated with the release.

## Purity Boundary

The corpus ledger in `corpus/ledger.json` is the admission gate. v0 code should
only train on generated curriculum derived from files listed there. Evaluation
probes are checked into the project, but they are source probes rather than a
claim of out-of-distribution generalization.

## Shape Of The Experiment

```text
corpus/glossary.json
corpus/grammar.json
corpus/admissions.jsonl
        |
        v
closed_world_lm.curriculum
        |
        v
build/train.txt + build/valid.txt
        |
        v
closed_world_lm.train
        |
        v
runs/<name>/checkpoint.json
        |
        v
closed_world_lm.evaluate
```

## Next Milestones

1. Complete the Python package/import migration from `closed_world_lm` to the
   QuarkLM naming plan without breaking existing run artifacts.
2. Expand the Docusaurus docs and standalone marketing page as the model,
   corpus, and release evidence grow.
3. Deepen self-diagnosis from explicit rules toward admitted-corpus-trained
   repair proposal and selection, without external model shaping.
4. Add larger continual-learning batches using generated probes and forgetting
   checks.
5. Strengthen prompt-conditioned representation so the direct transformer emits
   target-specific answers instead of a short global wrong answer, while
   preserving the `37/219` candidate discrimination and v0.42 target-loss gains.
6. Consider a from-scratch corpus-derived subword tokenizer only after the
   character-token transformer evidence shows tokenizer length is the bottleneck.
7. Fold the reliable decoder behavior back into the broader free-form character
   language model.
