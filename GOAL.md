# Goal Framework

## Objective

Continually improve QuarkLM, a closed-world learner that starts from no
pretrained weights or tokenizer, expands through an admitted model corpus,
and becomes reliable at responding to that corpus.

North star: build toward the world's first language model trained exclusively on
its own admitted dataset. QuarkLM is the human-facing product name, with
`quark-lm` as the intended repository/package slug. This is an aspiration, not
a completed claim.
Self-improvement applies to every part of the system: model weights, training
code, curriculum design, corpus quality, eval coverage, response reliability,
provenance, and the goal framework itself.

## Guardrails

- No pretrained weights.
- No pretrained tokenizer.
- No external embeddings.
- No unledgered training text.
- Any future tokenizer upgrade must be trained only from admitted corpus text;
  pretrained vocabularies are outside the boundary.
- Every run must record an untrained baseline and trained metrics.
- Every reliability claim needs an evaluation artifact.
- Improvements must preserve dataset exclusivity: new training examples must be
  generated from, or explicitly added to, the admitted corpus.
- Weight updates are part of the goal, but only as versioned training artifacts:
  no silent in-place mutation, no outside weights, no unmeasured promotion.
- Failed training or promotion attempts must remain archived evidence, not
  overwritten by later repair attempts.
- "Self" means operational self-knowledge only: the model may know its corpus
  boundary, weight-update process, admitted dataset, unknown policy, and current
  learning loop. It must not claim consciousness or subjective experience.
- "I learned something new" means the information was admitted into the
  ledgered corpus, converted into training lessons, used in a measured weight
  update, and preserved as auditable evidence.
- Code quality is part of self-improvement: use SOLID-aligned Python module
  boundaries, focused tests, clear artifacts, and small pure functions for
  corpus transforms, audits, feature extraction, and reporting.
- Public documentation is part of self-improvement: README, Docusaurus docs,
  and marketing pages must update with promoted releases whenever they describe
  current product state, commands, evals, evidence, or roadmap commitments.
- Eventual self-improvement should not depend on an external model shaping the
  learner. Near-term guidance may use deterministic, auditable rules over
  QuarkLM's own reports; the long-term target is repair proposal and selection
  learned from admitted artifacts and versioned outcomes.

## Current Loop

1. Expand `corpus/glossary.json` and `corpus/grammar.json`.
2. Regenerate `build/train.txt`, `build/valid.txt`, and `build/manifest.json`.
3. Train from random initialization.
4. Evaluate known, unknown, and held-out probes.
5. Compare target loss, free-form exact match, and closed-world candidate accuracy.
6. Promote curriculum/model changes only when the metrics move in the right direction.
7. When the model fails, improve the weakest part of the loop: corpus, lessons,
   architecture, optimizer, evals, or provenance.
8. Prefer repeatable self-improvement cycles over one-off commands, so progress
   leaves an auditable report artifact.
9. Admit new memory through `corpus/admissions.jsonl` before training; after
   admission, regenerate lessons and update weights so the model can truthfully
   answer that the new fact is part of its training data.
10. Generate direct and paraphrase admission probes from the admitted-memory log
    and audit probe sync in every self-improvement report.
11. Generate glossary definition probes from admitted glossary probe words and
    audit probe sync in every self-improvement report.
12. Enforce a promotion gate so a self-improvement command fails unless
    generated probes, prompt leakage, forgetting, and exact eval audits all pass.
13. Generate a self-diagnosis from the run report so the next improvement action
    is explicit, auditable, and labeled as using or not using an external model.
14. Archive each self-improvement attempt before updating the latest run report.
15. Train architecture prototypes, including the transformer learner, only from
    the admitted corpus and record their baseline/final metrics separately until
    they earn a promotion-gate role.
16. Keep public-facing docs and marketing aligned with the promoted state. The
    Docusaurus docs site targets `docs.quark-lm.eidetic-labs.com`; the
    standalone static marketing page targets `quark-lm.eidetic-labs.com`; both
    have GitHub Actions deployment scaffolds.

## Weight Update Policy

Weights must improve as the project improves. Each learned component should
start from random initialization unless an explicit admitted checkpoint is being
continued. A weight update is acceptable only when:

1. The training data comes from the admitted corpus or corpus-derived lessons.
2. The run records its seed, config, dataset source, baseline metrics, final
   metrics, and checkpoint path.
3. The updated checkpoint is kept as a versioned artifact under `runs/`.
4. The update is compared against the prior baseline before being treated as an
   improvement.
5. Failed or regressive checkpoints remain evidence, but are not promoted.
6. Held-out fact probes must not be trained with their exact evaluation prompt
   forms; they may only enter learned responders through admitted fact-style
   lessons.
7. QA-training facts may include both question-style and fact-style lessons so
   the model can learn transfer between surface forms without leaking held-out
   prompt answers.
8. The training model may add structural inductive bias, such as weighted
   semantic intent/entity features, when evaluation shows the learner is
   overfitting to surface form instead of using admitted facts.
9. Generated bridge lessons may be added from admitted facts when they improve
   transfer, but protected held-out evaluation prompts must remain absent from
   lesson files.
10. A promoted self-improvement run must pass the recorded promotion gate.
11. Experimental architecture checkpoints, such as the transformer learner,
    should be treated as evidence only for the behavior they actually show. A
    lower language-model loss is not a reliable-answer claim until answer evals
    pass.

## Reliability Strategy

The neural character model is the closed-world learner. The corpus response model is
the reliability rail: it learns a tiny fact table from admitted `fact:` lines and
answers source-grounded questions exactly. The response model gives us a teacher
and an oracle while the neural model learns to produce those answers directly.

The learned answer model is the bridge between those two layers: it starts with
random softmax weights, trains only on corpus-derived question/answer lessons,
and is evaluated on exact closed-world answers. It should become the first
learned component that reliably responds to the corpus before the free-form
character generator can do so.

The learned answer decoder moves that bridge closer to language modeling: it
generates answer characters one by one from prompt-conditioned weights instead
of choosing a whole answer label. It is still bounded and small, but it is a
generative checkpoint trained exclusively on admitted lessons.

The self-diagnosis layer is the current bridge toward autonomous improvement.
It reads QuarkLM's own run report and recommends the next repair or promotion
action with deterministic rules. The report records `uses_external_model: false`.
Future versions should move that diagnosis-and-repair policy into admitted,
trainable artifacts while preserving the same evidence boundary.

The tiny transformer learner is the first architecture path toward a more
recognizable language model. It starts from random weights, uses QuarkLM's
corpus-trained character tokenizer, and trains with a dependency-free scalar
autodiff engine. It is not yet the reliable response path; it is an auditable
experimental backend that must mature under the same corpus and eval gates.

## Engineering Quality Strategy

The codebase should stay small but intentionally shaped. `self_improve.py`
orchestrates runs; corpus provenance lives in `provenance.py`; admission,
curriculum, response, learned answer selection, and generative decoding each
own their own behavior. Tests should cover both model behavior and supporting
infrastructure such as admission rejection, forgetting audits, weighted training
pools, and corpus diffs.

## Latest Evidence

`runs/context64-v0.2/`:

- validation NLL: `3.4968 -> 2.6545`
- known QA target NLL: `3.4979 -> 2.4155`
- held-out target NLL: `3.4978 -> 2.5788`
- free-form exact match is still `0`

`runs/answer-v0.1/`:

- learned answer model baseline exact rates: QA `1/8`, unknown `0/4`,
  held-out `1/8`, paraphrase `0/8`
- trained exact rates before unseen-paraphrase tightening: QA `8/8`, unknown
  `4/4`, held-out `8/8`, paraphrase `8/8`

`runs/answer-v0.2/`:

- learned answer model with stricter unseen paraphrase probes
- baseline exact rates: QA `1/8`, unknown `0/4`, held-out `1/8`,
  paraphrase `0/8`
- trained exact rates: QA `8/8`, unknown `4/4`, held-out `8/8`,
  paraphrase `8/8`

`runs/decoder-v0.2/`:

- generative answer decoder from random weights
- baseline exact rates: QA `0/8`, unknown `0/4`, held-out `0/8`,
  paraphrase `0/8`
- trained exact rates: QA `8/8`, unknown `4/4`, held-out `8/8`,
  paraphrase `8/8`

`runs/self-improve-v0.9/`:

- stricter lesson split: held-out facts train only through fact-style lessons,
  not exact held-out evaluation prompts
- prompt leakage audit: passed, with zero leaked held-out prompts
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`

`runs/self-improve-v0.12/`:

- operational self and learning-admission concepts added to the admitted corpus
- one admitted memory event: `learned-teacher-tree`
- prompt leakage audit: passed, with zero leaked held-out prompts
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `4/4`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `4/4`
- decoder trainer now records a weighted training pool, so longer and core
  self-improvement lessons get enough updates while preserving the original
  lesson corpus

`runs/self-improve-v0.14/`:

- continual admission batch expanded the admitted memory log to two facts:
  `learned-teacher-tree` and `learned-child-bag`
- admission probes expanded from `4` to `8`
- forgetting audit compared against `runs/self-improve-v0.12/` and passed
- prompt leakage audit: passed, with zero leaked held-out prompts
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `8/8`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `8/8`
- answer model and decoder now record weighted training-pool sizes so
  anti-forgetting pressure is visible in the report

`runs/self-improve-v0.16/`:

- SOLID-aligned provenance code moved into `closed_world_lm.provenance`
- run writes `corpus_snapshot.json` and `corpus_diff.json`
- corpus diff compared against `runs/self-improve-v0.15/` and evaluated all
  ledgered sources as unchanged
- forgetting audit compared against `runs/self-improve-v0.15/` and passed
- prompt leakage audit: passed, with zero leaked held-out prompts
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `8/8`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `8/8`

`runs/self-improve-v0.17/`:

- admission probes are generated from `corpus/admissions.jsonl`
- `closed_world_lm.admission_probes --check` passes with `8` expected records,
  `8` actual records, and zero missing, extra, or mismatched ids
- run report includes `admission_probe_audit`, which passed
- corpus diff compared against `runs/self-improve-v0.16/` and evaluated all
  ledgered sources as unchanged
- forgetting audit compared against `runs/self-improve-v0.16/` and passed
- prompt leakage audit: passed, with zero leaked held-out prompts
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `8/8`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `8/8`

`runs/self-improve-v0.18/`:

- human-facing product name is now QuarkLM, with `quark-lm` as the package and
  intended GitHub slug, and "Big idea. Tiny package." as the working tagline
- `quark-lm-*` console-script aliases were added while `closed_world_lm`
  remains the current Python import path
- admission paraphrase probes are generated from `corpus/admissions.jsonl`
- `closed_world_lm.admission_probes --check` passes direct probes `8/8` and
  paraphrase probes `14/14`, with zero missing, extra, or mismatched ids
- run report includes direct and paraphrase `admission_probe_audit`, which
  passed
- corpus diff compared against `runs/self-improve-v0.17/` and added
  `admission-paraphrase-probes-v0` as an eval-only ledger source
- forgetting audit compared against `runs/self-improve-v0.17/` and passed
- prompt leakage audit: passed, with zero leaked held-out prompts
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `8/8`, admission paraphrases `14/14`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `8/8`, admission paraphrases `14/14`

`runs/self-improve-v0.19/`:

- added glossary word `stone` and admitted `learned-ivy-stone` before training
- admission probe sync expanded direct probes to `12/12` and paraphrase probes
  to `21/21`
- admitted memories now generate direct QA lessons plus fact bridge lessons
- fact bridge examples are upweighted to preserve held-out question transfer
  without adding protected held-out prompts to training lessons
- learned-component eval summaries now retain `failed_records` for diagnosis
- corpus diff compared against `runs/self-improve-v0.18/` and recorded changed
  glossary, admissions, direct admission probes, and admission paraphrase probes
- forgetting audit compared against `runs/self-improve-v0.18/` and passed
- prompt leakage audit: passed, with zero leaked held-out prompts
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `12/12`, admission paraphrases `21/21`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `12/12`, admission paraphrases `21/21`

`runs/self-improve-v0.20/`:

- glossary probes are generated from `corpus/glossary.json` probe words
- `closed_world_lm.glossary_probes --check` passes `20/20`, with zero missing,
  extra, or mismatched ids
- run report includes `glossary_probe_audit`, `exact_eval_audit`, and
  `promotion_gate`, all passed
- generated bridge lessons add non-held-out transfer prompts such as
  `tell me the place of eli lamp`
- protected prompt leakage audit: passed, with zero leaked held-out prompts
- corpus diff compared against `runs/self-improve-v0.19/` and recorded
  `glossary-probes-v0` as an added eval-only ledger source plus a changed
  glossary source
- forgetting audit compared against `runs/self-improve-v0.19/` and passed
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `12/12`, admission paraphrases `21/21`,
  glossary `20/20`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `12/12`, admission paraphrases `21/21`,
  glossary `20/20`

`runs/self-improve-v0.21/`:

- added glossary probe words `shell`, `coin`, and `drum`
- admitted three new memories: `learned-noah-shell`, `learned-ava-coin`, and
  `learned-omar-drum`
- admission probe sync expanded direct probes to `24/24` and paraphrase probes
  to `42/42`
- glossary probes expanded to `26/26`
- unknown bridge lessons now include generated unknown fact prompts such as
  `tell me the place of noah ball` so admitted facts do not overgeneralize into
  unknown paraphrases
- answer classifier self and learning lessons were rebalanced to preserve
  operational self-knowledge as the corpus grew
- run report includes rule-based `self_diagnosis` with
  `uses_external_model: false`, zero blockers, and recommendation
  `promote_or_expand_corpus`
- corpus diff compared against `runs/self-improve-v0.20/` and recorded added
  admissions plus changed glossary, admission, admission-paraphrase, and
  glossary-probe sources
- forgetting audit compared against `runs/self-improve-v0.20/` and passed
- protected prompt leakage audit: passed, with zero leaked held-out prompts
- exact eval audit and promotion gate passed
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `24/24`, admission paraphrases `42/42`,
  glossary `26/26`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `5/5`,
  learning `3/3`, admissions `24/24`, admission paraphrases `42/42`,
  glossary `26/26`

`runs/self-improve-v0.22/`:

- expanded operational self facts from `5` to `7`
- expanded learning rules from `3` to `4`
- added self-diagnosis corpus facts: diagnosis is guided by self-improvement
  reports, and no external model shapes it
- added learning rule: the next repair action is chosen from report evidence
- first v0.22 attempt failed the promotion gate and exposed that top-level run
  reports alone were not enough to preserve failed-attempt evidence
- the embedded `self_diagnosis` recommended strengthening non-held-out bridge
  transfer plus decoder self/learning and glossary rebalancing
- repaired training pools according to that diagnosis, then reran the same
  promotion gate without weakening requirements
- corpus diff compared against `runs/self-improve-v0.21/` and recorded changed
  grammar, self-probe, and learning-probe sources; admissions and generated
  admission/glossary probes were unchanged
- forgetting audit compared against `runs/self-improve-v0.21/` and passed
- protected prompt leakage audit: passed, with zero leaked held-out prompts
- exact eval audit and promotion gate passed
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `24/24`, admission paraphrases `42/42`,
  glossary `26/26`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `24/24`, admission paraphrases `42/42`,
  glossary `26/26`
- final `self_diagnosis` records `uses_external_model: false`, zero blockers,
  and recommendation `promote_or_expand_corpus`

`runs/self-improve-v0.23/`:

- added attempt archives under `runs/self-improve-v0.23/attempts/`
- intentionally undertrained `attempt-001` failed the promotion gate and remains
  preserved as failed evidence with `278` diagnosis blockers
- repaired `attempt-002` passed the same promotion gate with zero blockers
- top-level `runs/self-improve-v0.23/self_improvement_report.json` now points
  at the latest attempt while the failed report remains intact
- checkpoints for answer classifier and decoder are written inside the attempt
  directory so weights and reports share provenance

`runs/self-improve-v0.24/`:

- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.23/` and passed
- protected prompt leakage, exact eval audit, and promotion gate passed
- standalone `self_diagnosis.json` records `uses_external_model: false`, zero
  blockers, and recommendation `promote_or_expand_corpus`
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `24/24`, admission paraphrases `42/42`,
  glossary `26/26`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `24/24`, admission paraphrases `42/42`,
  glossary `26/26`

`runs/transformer-v0.24/`:

- first dependency-free tiny decoder-only transformer checkpoint:
  `runs/transformer-v0.24/transformer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- trained for `40` steps from random initialization
- validation NLL moved from `3.5949` to `3.4779`
- answer exact eval is still `0/28`, so this is architecture evidence, not a
  reliable-response promotion

`runs/self-improve-v0.25/`:

- admitted three new memories before training:
  `learned-mia-ring`, `learned-leo-kite`, and `learned-nina-bell`
- expanded glossary probe words with `ring`, `kite`, and `bell`
- admitted facts expanded from `6` to `9`
- admission probe sync expanded direct probes to `36/36` and paraphrase probes
  to `63/63`
- glossary probes expanded to `32/32`
- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.24/` and passed
- protected prompt leakage, exact eval audit, and promotion gate passed
- standalone `self_diagnosis.json` records `uses_external_model: false`, zero
  blockers, and recommendation `promote_or_expand_corpus`
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `36/36`, admission paraphrases `63/63`,
  glossary `32/32`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `36/36`, admission paraphrases `63/63`,
  glossary `32/32`

`runs/transformer-v0.25/`:

- current dependency-free tiny decoder-only transformer checkpoint:
  `runs/transformer-v0.25/transformer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- trained for `40` steps from random initialization on the expanded corpus
- validation NLL moved from `3.5885` to `3.4382`
- answer exact eval is still `0/28`, so this remains architecture evidence, not
  a reliable-response promotion

`runs/self-improve-v0.26/`:

- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.25/` and passed
- protected prompt leakage, exact eval audit, and promotion gate passed
- standalone `self_diagnosis.json` records `uses_external_model: false`, zero
  blockers, and recommendation `promote_or_expand_corpus`
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `36/36`, admission paraphrases `63/63`,
  glossary `32/32`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `36/36`, admission paraphrases `63/63`,
  glossary `32/32`

`runs/transformer-answer-v0.26/`:

- added transformer `answer-train` path using corpus-derived `AnswerExample`
  lessons
- checkpoint: `runs/transformer-answer-v0.26/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- trained for `400` answer-lesson steps from random initialization
- average answer target NLL moved from `3.5730` to `2.6781`
- exact answer eval remains `0/180`
- broad candidate accuracy moved from `18/180` to `2/180`; the context-16 run
  over-favored short answers such as `no.`, so candidate discrimination is not
  solved
- a context-48 repair attempt was interrupted during full-suite scoring because
  the current evaluator is too slow at that context size

`runs/self-improve-v0.27/`:

- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.26/` and passed
- protected prompt leakage, exact eval audit, and promotion gate passed
- standalone `self_diagnosis.json` records `uses_external_model: false`, zero
  blockers, and recommendation `promote_or_expand_corpus`
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `36/36`, admission paraphrases `63/63`,
  glossary `32/32`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `36/36`, admission paraphrases `63/63`,
  glossary `32/32`

`runs/transformer-answer-v0.27/`:

- added a faster transformer answer evaluator that can use only active eval-set
  candidates and skip free-form completions
- checkpoint: `runs/transformer-answer-v0.27/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- trained for `200` answer-lesson steps from random initialization
- context size increased to `48`
- average answer target NLL moved from `3.5878` to `2.9594`
- exact answer generation was not measured by the fast candidate-only path
- eval-scoped candidate accuracy stayed `16/180 -> 16/180`; larger context and
  faster scoring improved measurement and loss, but did not improve answer
  selection

`runs/transformer-answer-v0.28-choice-prefix-pilot/`:

- added candidate-discriminative answer-training knobs:
  `--choice-loss-weight`, `--choice-negatives`, and `--choice-max-chars`
- full-sequence candidate contrast was interrupted because the scalar
  transformer path was too slow for that training shape
- prefix-choice pilot checkpoint:
  `runs/transformer-answer-v0.28-choice-prefix-pilot/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- trained for `80` answer-lesson steps from random initialization
- context size stayed at `48`
- average answer target NLL moved from `3.5878` to `3.1578`
- exact answer generation was not measured by the fast candidate-only path
- eval-scoped candidate accuracy stayed `16/180 -> 16/180`; candidate
  discrimination remains unsolved

`runs/self-improve-v0.28/`:

- admitted memories `learned-sara-marble`, `learned-milo-spoon`, and
  `learned-ruth-ribbon`
- added glossary probe words `marble`, `spoon`, and `ribbon`
- attempt `001` failed because the answer decoder regressed on
  `learning-weight-update`
- attempt `002` overcorrected learning/self lesson weight and regressed place
  answers
- answer decoder learning/self repeat weight settled at
  `DECODER_SELF_LEARNING_REPEATS = 55`
- standard self-improvement cycle passed on archived `attempt-003`
- forgetting audit compared against `runs/self-improve-v0.27/` and passed
- protected prompt leakage, exact eval audit, and promotion gate passed
- standalone `self_diagnosis.json` records `uses_external_model: false`, zero
  blockers, and recommendation `promote_or_expand_corpus`
- generated probe counts: admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`

`runs/transformer-answer-v0.29-selector-fast/`:

- added `AnswerCandidateSelector`, a small closed-world selector paired with
  transformer answer evidence
- selector weights start from random initialization and train only on
  corpus-derived `AnswerExample` lessons
- selector checkpoint:
  `runs/transformer-answer-v0.29-selector-fast/answer_selector.json`
- transformer checkpoint:
  `runs/transformer-answer-v0.29-selector-fast/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- transformer trained for `80` target-loss answer-lesson steps from random
  initialization
- context size stayed at `48`
- average answer target NLL moved from `3.5894` to `3.2662`
- exact answer generation was not measured by the fast candidate-only path
- transformer-only eval-scoped candidate accuracy stayed `16/219 -> 16/219`
- selector trained for `1600` steps and moved eval-scoped candidate accuracy
  from `18/219 -> 219/219`
- v0.29 resolves candidate choice for an auxiliary closed-world selector, but
  it does not yet make the transformer a reliable free-form answer generator

`runs/self-improve-v0.29/`:

- kept corpus sources unchanged from v0.28 at `12` admitted facts
- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.28/` and passed
- protected prompt leakage, exact eval audit, and promotion gate passed
- report and standalone `self_diagnosis.json` record `uses_external_model:
  false`, zero blockers, and recommendation `promote_or_expand_corpus`
- generated probe counts: admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`

`runs/transformer-answer-v0.30-selector-emission/`:

- added explicit selector-assisted answer emission to the transformer answer
  experiment
- new `--selector-emit-completions` flag records the selector's chosen
  closed-world candidate as the emitted completion for exact-match evidence
- selector checkpoint:
  `runs/transformer-answer-v0.30-selector-emission/answer_selector.json`
- transformer checkpoint:
  `runs/transformer-answer-v0.30-selector-emission/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- transformer trained for `80` target-loss answer-lesson steps from random
  initialization
- context size stayed at `48`
- average answer target NLL moved from `3.5894` to `3.2662`
- free-form transformer generation was not measured by this fast path
- transformer-only eval-scoped candidate accuracy stayed `16/219 -> 16/219`
- selector trained for `1600` steps and moved both eval-scoped candidate
  accuracy and emitted exact answers from `18/219 -> 219/219`
- v0.30 resolves closed-world candidate emission for the auxiliary selector,
  but it does not yet make the transformer a reliable free-form answer
  generator

`runs/self-improve-v0.30/`:

- kept corpus sources unchanged from v0.29 at `12` admitted facts
- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.29/` and passed
- protected prompt leakage, exact eval audit, and promotion gate passed
- report and standalone `self_diagnosis.json` record `uses_external_model:
  false`, zero blockers, and recommendation `promote_or_expand_corpus`
- generated probe counts: admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`

`runs/transformer-answer-v0.31-generator-weighted-lr035-80k/`:

- added `TransformerGuidedAnswerGenerator`, a character answer generator paired
  with transformer answer evidence
- generator weights start from random initialization and train only on
  corpus-derived `AnswerExample` lessons
- generator features include prompt/prefix features plus top corpus-token
  predictions from the trained transformer
- added cached generator lessons so long generator runs do not recompute
  transformer prediction features on every update
- added a generator-specific weighted training pool so long self, learning, and
  glossary answers receive enough updates
- generator checkpoint:
  `runs/transformer-answer-v0.31-generator-weighted-lr035-80k/answer_generator.json`
- selector checkpoint:
  `runs/transformer-answer-v0.31-generator-weighted-lr035-80k/answer_selector.json`
- transformer checkpoint:
  `runs/transformer-answer-v0.31-generator-weighted-lr035-80k/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- transformer trained for `80` target-loss answer-lesson steps from random
  initialization
- context size stayed at `48`
- average answer target NLL moved from `3.5894` to `3.2662`
- transformer-only eval-scoped candidate accuracy stayed `16/219 -> 16/219`
- selector candidate/emitted exact evidence stayed `18/219 -> 219/219`
- no-candidate generator trained for `80000` weighted steps at learning rate
  `0.035`
- generator exact answers moved `0/219 -> 219/219`
- generator weighted target loss moved `3.3160 -> 0.0029`
- `uses_answer_candidates: false`
- v0.31 resolves answer generation without a closed candidate set for the
  auxiliary transformer-guided generator, but raw transformer decoding itself
  still does not generate exact answers

`runs/self-improve-v0.31/`:

- kept corpus sources unchanged from v0.30 at `12` admitted facts
- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.30/` and passed
- protected prompt leakage, exact eval audit, and promotion gate passed
- report and standalone `self_diagnosis.json` record `uses_external_model:
  false`, zero blockers, and recommendation `promote_or_expand_corpus`
- generated probe counts: admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`

`runs/transformer-answer-v0.32-direct-base-context32/`:

- added stop-aware greedy generation to the tiny transformer so direct answer
  completion can stop on an admitted newline terminator
- added cached direct-answer lessons that train the transformer's own weights
  rather than an auxiliary answer generator
- added direct-answer metrics with strict exact matching, target loss,
  `uses_answer_candidates: false`, and `auxiliary_weights: false`
- added `--direct-answer-steps`, `--direct-answer-learning-rate`,
  `--direct-answer-eval-every`, `--direct-answer-max-new-chars`,
  `--direct-answer-mode`, and `--direct-answer-terminator` to
  `transformer_char_model answer-train`
- added `first-error` direct-answer repair mode, which finds the current greedy
  mismatch and trains that transformer context; the selected v0.32 evidence
  uses `random-char` mode because the first-error pilot overcorrected into
  repeated `" a"` completions
- checkpoint: `runs/transformer-answer-v0.32-direct-base-context32/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- transformer trained for `80` target-loss answer-lesson steps from random
  initialization
- direct answer phase trained for `1000` steps on `9144` weighted direct-answer
  examples
- context size was `32`
- average transformer answer target NLL moved from `3.5828` to `2.9451`
- direct answer target loss moved from `3.3496` to `2.6780`
- transformer-only eval-scoped candidate accuracy moved `15/219 -> 16/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- current failure mode is repeated-character completion, e.g. repeated `"n"`
  after the answer prompt
- v0.32 moves answer training into the transformer weights themselves, but it
  does not yet solve raw transformer decoding

`runs/self-improve-v0.32/`:

- kept corpus sources unchanged from v0.31 at `12` admitted facts
- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.31/` and passed
- protected prompt leakage, exact eval audit, and promotion gate passed
- report and standalone `self_diagnosis.json` record `uses_external_model:
  false`, zero blockers, and recommendation `promote_or_expand_corpus`
- generated probe counts: admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`

`runs/transformer-answer-v0.33-unlikelihood-context32/`:

- added transformer `train_step_with_unlikelihood`, a direct weight update that
  trains the target token and penalizes one wrong self-predicted token
- added `first-error-unlikelihood` direct-answer mode
- added `--direct-answer-negative-weight`
- the wrong token is produced by QuarkLM's own greedy prediction on an admitted
  lesson; no external model, candidate set, or auxiliary answer weights are used
- checkpoint:
  `runs/transformer-answer-v0.33-unlikelihood-context32/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- transformer trained for `80` target-loss answer-lesson steps from random
  initialization
- direct answer phase trained for `1000` steps on `9144` weighted direct-answer
  examples
- direct answer mode was `first-error-unlikelihood`
- negative weight was `1.0`
- context size was `32`
- average transformer answer target NLL moved from `3.5828` to `2.9504`
- direct answer target loss moved from `3.3496` to `3.1536`
- transformer-only eval-scoped candidate accuracy moved `15/219 -> 37/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- current failure mode changed from repeated `"n"` in v0.32 to repeated
  `" a"` in v0.33
- v0.33 improves direct transformer discrimination from self-generated errors,
  but it does not yet solve raw greedy answer emission

`runs/self-improve-v0.33/`:

- kept corpus sources unchanged from v0.32 at `12` admitted facts
- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.32/` and passed
- protected prompt leakage, exact eval audit, and promotion gate passed
- report and standalone `self_diagnosis.json` record `uses_external_model:
  false`, zero blockers, and recommendation `promote_or_expand_corpus`
- generated probe counts: admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`

`runs/transformer-answer-v0.34-staged-context32/`:

- added `direct_answer_rollout_error`, which evaluates direct answer repair on
  the transformer's own generated prefix rather than only teacher-forced target
  prefixes
- added `rollout-unlikelihood`, `hybrid-unlikelihood`, and
  `staged-unlikelihood` direct-answer modes
- selected evidence uses `staged-unlikelihood`: first half first-error
  unlikelihood, second half rollout unlikelihood
- checkpoint:
  `runs/transformer-answer-v0.34-staged-context32/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- transformer trained for `80` target-loss answer-lesson steps from random
  initialization
- direct answer phase trained for `1000` steps on `9144` weighted direct-answer
  examples
- direct answer mode was `staged-unlikelihood`
- negative weight was `1.0`
- context size was `32`
- average transformer answer target NLL moved from `3.5828` to `2.9804`
- direct answer target loss moved from `3.3496` to `2.8851`
- transformer-only eval-scoped candidate accuracy moved `15/219 -> 27/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- current failure mode changed from v0.33's repeated `" a"` loop to shorter
  `" ae"` style loops
- v0.34 improves generated-prefix exposure training and direct loss, but it
  does not yet solve raw greedy answer emission
- v0.33 remains the strongest transformer-only candidate run so far at
  `15/219 -> 37/219`

`runs/self-improve-v0.34/`:

- kept corpus sources unchanged from v0.33 at `12` admitted facts
- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.33/` and passed
- protected prompt leakage, exact eval audit, and promotion gate passed
- report and standalone `self_diagnosis.json` record `uses_external_model:
  false`, zero blockers, and recommendation `promote_or_expand_corpus`
- generated probe counts: admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`

`runs/transformer-answer-v0.35-periodic10-context32/`:

- added `periodic-rollout-unlikelihood` direct-answer mode
- added `--direct-answer-rollout-interval`
- selected evidence uses first-error unlikelihood for most updates and rollout
  unlikelihood every `10` direct-answer steps
- checkpoint:
  `runs/transformer-answer-v0.35-periodic10-context32/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- transformer trained for `80` target-loss answer-lesson steps from random
  initialization
- direct answer phase trained for `1000` steps on `9144` weighted direct-answer
  examples
- direct answer mode was `periodic-rollout-unlikelihood`
- negative weight was `1.0`
- rollout interval was `10`
- context size was `32`
- average transformer answer target NLL moved from `3.5828` to `3.0443`
- direct answer target loss moved from `3.3496` to `2.9526`
- transformer-only eval-scoped candidate accuracy moved `15/219 -> 37/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- current failure mode changed from v0.34's shorter `" ae"` loop to premature
  `" t"` stopping
- v0.35 recovers v0.33's candidate-discrimination gain while keeping a
  generated-prefix rollout repair signal, but it does not yet solve raw greedy
  answer emission

`runs/self-improve-v0.35/`:

- kept corpus sources unchanged from v0.34 at `12` admitted facts
- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.34/` and passed
- protected prompt leakage, exact eval audit, and promotion gate passed
- report and standalone `self_diagnosis.json` record `uses_external_model:
  false`, zero blockers, and recommendation `promote_or_expand_corpus`
- generated probe counts: admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`

`runs/transformer-answer-v0.36-periodic-earlystop10-context32/`:

- added `direct_answer_early_stop_error`
- added `train_direct_answer_early_stop_unlikelihood`
- added `early-stop-unlikelihood` and `periodic-early-stop-unlikelihood`
  direct-answer modes
- selected evidence uses first-error unlikelihood for most updates and
  early-stop unlikelihood every `10` direct-answer steps
- checkpoint:
  `runs/transformer-answer-v0.36-periodic-earlystop10-context32/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- transformer trained for `80` target-loss answer-lesson steps from random
  initialization
- direct answer phase trained for `1000` steps on `9144` weighted direct-answer
  examples
- direct answer mode was `periodic-early-stop-unlikelihood`
- negative weight was `1.0`
- early-stop repair interval was `10`
- context size was `32`
- average transformer answer target NLL moved from `3.5828` to `2.9311`
- direct answer target loss moved from `3.3496` to `3.1536`
- transformer-only eval-scoped candidate accuracy moved `15/219 -> 37/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- current failure mode changed from v0.35's premature `" t"` stop to repeated
  `" a"` loops
- v0.36 preserves the best candidate-discrimination gain and proves the
  terminator can be targeted from the model's own generated failure, but it does
  not yet solve raw greedy answer emission

`runs/self-improve-v0.36/`:

- kept corpus sources unchanged from v0.35 at `12` admitted facts
- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.35/` and passed
- protected prompt leakage, exact eval audit, and promotion gate passed
- report and standalone `self_diagnosis.json` record `uses_external_model:
  false`, zero blockers, and recommendation `promote_or_expand_corpus`
- generated probe counts: admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`

`runs/transformer-answer-v0.37-periodic-repeat10-context32/`:

- added `has_repeated_suffix`
- added `direct_answer_repeat_loop_error`
- added `train_direct_answer_repeat_loop_unlikelihood`
- added `repeat-loop-unlikelihood` and `periodic-repeat-loop-unlikelihood`
  direct-answer modes
- this run used repeat-loop repair every `10` direct-answer steps
- checkpoint:
  `runs/transformer-answer-v0.37-periodic-repeat10-context32/transformer_answer.json`
- average transformer answer target NLL moved from `3.5828` to `3.0723`
- direct answer target loss moved from `3.3496` to `3.3070`
- transformer-only eval-scoped candidate accuracy moved `15/219 -> 17/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- current failure mode changed to repeated `"tnnn"` style loops
- this attempt is not the selected v0.37 evidence because it damaged candidate
  discrimination

`runs/transformer-answer-v0.37-periodic-repeat50-context32/`:

- selected evidence uses first-error unlikelihood for most updates and
  repeat-loop unlikelihood every `50` direct-answer steps
- checkpoint:
  `runs/transformer-answer-v0.37-periodic-repeat50-context32/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- transformer trained for `80` target-loss answer-lesson steps from random
  initialization
- direct answer phase trained for `1000` steps on `9144` weighted direct-answer
  examples
- direct answer mode was `periodic-repeat-loop-unlikelihood`
- negative weight was `1.0`
- repeat-loop repair interval was `50`
- context size was `32`
- average transformer answer target NLL moved from `3.5828` to `2.9041`
- direct answer target loss moved from `3.3496` to `3.0929`
- transformer-only eval-scoped candidate accuracy moved `15/219 -> 37/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- current failure mode changed from v0.36's repeated `" a"` loop to repeated
  `" t"` loops
- v0.37 preserves the best candidate-discrimination gain and improves scored
  target losses, but it proves that unlikelihood-only symptom repair still
  moves the greedy failure rather than solving raw answer emission

`runs/self-improve-v0.37/`:

- kept corpus sources unchanged from v0.36 at `12` admitted facts
- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.36/` and passed
- protected prompt leakage, exact eval audit, and promotion gate passed
- report and standalone `self_diagnosis.json` record `uses_external_model:
  false`, zero blockers, and recommendation `promote_or_expand_corpus`
- generated probe counts: admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`

`runs/transformer-answer-v0.38-periodic-balanced50-context32/`:

- added `train_step_with_unlikelihood_and_positive`
- added `direct_answer_balanced_repair_error`
- added `train_direct_answer_balanced_repair_unlikelihood`
- added `balanced-repair-unlikelihood` and
  `periodic-balanced-repair-unlikelihood` direct-answer modes
- added `--direct-answer-positive-weight`
- selected evidence uses first-error unlikelihood for most updates and balanced
  self-repair plus teacher-forced continuation every `50` direct-answer steps
- checkpoint:
  `runs/transformer-answer-v0.38-periodic-balanced50-context32/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- transformer trained for `80` target-loss answer-lesson steps from random
  initialization
- direct answer phase trained for `1000` steps on `9144` weighted direct-answer
  examples
- direct answer mode was `periodic-balanced-repair-unlikelihood`
- negative weight was `1.0`
- positive weight was `1.0`
- balanced repair interval was `50`
- context size was `32`
- average transformer answer target NLL moved from `3.5828` to `2.8552`
- direct answer target loss moved from `3.3496` to `3.0399`
- transformer-only eval-scoped candidate accuracy moved `15/219 -> 37/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- current failure mode remained repeated `" t"` loops
- v0.38 improves scored target distribution while preserving candidate
  discrimination, but raw greedy emission still needs sequence-level recovery
  training

`runs/transformer-answer-v0.38-periodic-balanced50-w2-context32/`:

- this run used the same balanced mode and interval as the selected run, but
  set positive weight to `2.0`
- checkpoint:
  `runs/transformer-answer-v0.38-periodic-balanced50-w2-context32/transformer_answer.json`
- average transformer answer target NLL moved from `3.5828` to `3.1227`
- direct answer target loss moved from `3.3496` to `3.2697`
- transformer-only eval-scoped candidate accuracy moved `15/219 -> 17/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- current failure mode changed to repeated `"tnnn"` style loops
- this attempt is not the selected v0.38 evidence because stronger positive
  pressure damaged candidate discrimination and target losses

`runs/self-improve-v0.38/`:

- kept corpus sources unchanged from v0.37 at `12` admitted facts
- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.37/` and passed
- protected prompt leakage, exact eval audit, and promotion gate passed
- report and standalone `self_diagnosis.json` record `uses_external_model:
  false`, zero blockers, and recommendation `promote_or_expand_corpus`
- generated probe counts: admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- learned answer model trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`
- generative answer decoder trained exact rates: QA `8/8`, unknown `4/4`,
  held-out `8/8`, paraphrase `8/8`, owner `10/10`, self `7/7`,
  learning `4/4`, admissions `48/48`, admission paraphrases `84/84`,
  glossary `38/38`

The next improvement target is adding sequence-level generated-prefix recovery
training so raw greedy decoding learns to continue after its own imperfect
prefixes while preserving the `37/219` candidate-discrimination gain and v0.38
target-loss gains; then continuing
admitted-memory batches, completing the Python package/import migration to
QuarkLM naming, turning more of the deterministic self-diagnosis and repair
policy into admitted-corpus-trained behavior, and folding the decoder's
reliability back into the broader free-form language model.
