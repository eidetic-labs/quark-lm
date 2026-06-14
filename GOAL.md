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
16. Audit prompt context coverage for transformer answer runs so the loop can
    distinguish missing input evidence from insufficient weight updates.
17. Keep public-facing docs and marketing aligned with the promoted state. The
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
12. A context-size increase should record prompt coverage, runtime cost, direct
    loss, answer NLL, exact greedy output, candidate accuracy, and failure
    pattern before it is considered a model improvement.
13. A bounded architecture screen may skip an expensive post-direct candidate
    snapshot only when the run metrics record that skip. Such a run can prove
    training-loop completion, checkpoint writing, and direct-answer loss
    movement, but it is not promotion evidence until a full final candidate
    evaluation is recorded.
14. Direct-answer snapshots should include branch profiles from QuarkLM's own
    logits so repair selection can distinguish prompt-independent branch
    collapse from later answer-generation failures without relying on an
    external model.

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

`runs/transformer-answer-v0.39-periodic-recovery50-context32/` and
`runs/transformer-answer-v0.39-periodic-recovery200-context32/`:

- added `direct_answer_generated_prefix_recovery`
- added `train_direct_answer_generated_prefix_recovery_unlikelihood`
- added `generated-prefix-recovery-unlikelihood` and
  `periodic-generated-prefix-recovery-unlikelihood` direct-answer modes
- added `--direct-answer-recovery-steps`
- both runs are preserved as rejected evidence
- interval `50` damaged candidate discrimination (`15/219 -> 17/219`) and
  ended in repeated `"afo"`-style loops
- interval `200` preserved candidate discrimination (`15/219 -> 37/219`) but
  produced worse direct loss and answer NLL than v0.38, ending in repeated
  `" a"` loops
- lesson: training after corrupted generated prefixes can move the failure loop
  without improving exact greedy answers or scored target distribution

`runs/transformer-answer-v0.39-periodic-sequence50-context32/`:

- added `direct_answer_sequence_repair_errors`
- added `train_direct_answer_sequence_repair_unlikelihood`
- added `sequence-repair-unlikelihood` and
  `periodic-sequence-repair-unlikelihood` direct-answer modes
- selected evidence keeps most direct updates on first-error unlikelihood and
  injects teacher-forced sequence repair every `50` direct-answer steps
- checkpoint:
  `runs/transformer-answer-v0.39-periodic-sequence50-context32/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- transformer trained for `80` target-loss answer-lesson steps from random
  initialization
- direct answer phase trained for `1000` steps on `9144` weighted direct-answer
  examples
- direct answer mode was `periodic-sequence-repair-unlikelihood`
- negative weight was `1.0`
- positive weight was `1.0`
- sequence repair interval was `50`
- context size was `32`
- average transformer answer target NLL moved from `3.5828` to `2.8257`
- direct answer target loss moved from `3.3496` to `2.9793`
- transformer-only eval-scoped candidate accuracy moved `15/219 -> 37/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- current failure mode remained repeated `" t"` loops
- v0.39 improves scored target distribution beyond v0.38 while preserving
  candidate discrimination, but raw greedy emission still needs repeat-loop
  escape training

`runs/self-improve-v0.39/`:

- kept corpus sources unchanged from v0.38 at `12` admitted facts
- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.38/` and passed
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

`runs/transformer-answer-v0.40-periodic-loop50-context32/`:

- added `train_direct_answer_loop_escape_unlikelihood`
- added `loop-escape-unlikelihood` and
  `periodic-loop-escape-unlikelihood` direct-answer modes
- loop escape pairs a generated repeated-loop penalty with a positive admitted
  continuation
- checkpoint:
  `runs/transformer-answer-v0.40-periodic-loop50-context32/transformer_answer.json`
- this run is preserved as rejected evidence
- direct answer target loss moved from `3.3496` to `3.3128`
- average transformer answer target NLL moved from `3.5828` to `3.1315`
- transformer-only eval-scoped candidate accuracy regressed `15/219 -> 17/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- failure mode changed from repeated `" t"` loops to repeated `"tnonono"`
  loops
- lesson: direct loop pressure at interval `50` changes the loop shape but
  damages scored target distribution and candidate discrimination

`runs/transformer-answer-v0.40-sequence50-loop200-context32/`:

- added `periodic-sequence-loop-escape-unlikelihood`
- added `--direct-answer-sequence-interval`
- this run kept v0.39's sequence-repair cadence and injected loop escape every
  `200` steps
- checkpoint:
  `runs/transformer-answer-v0.40-sequence50-loop200-context32/transformer_answer.json`
- this run is preserved as non-selected evidence
- direct answer target loss moved from `3.3496` to `2.9855`
- average transformer answer target NLL moved from `3.5828` to `2.7565`
- transformer-only eval-scoped candidate accuracy moved `15/219 -> 37/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- failure mode remained repeated `" t"` loops
- lesson: sparse loop escape can preserve candidate discrimination and improve
  NLL, but it does not solve greedy loop emission and trails the selected
  branch-repair loss

`runs/transformer-answer-v0.40-branch-context32/`:

- added `direct_answer_branch_repair_error`
- added `train_direct_answer_branch_repair_unlikelihood`
- added `branch-repair-unlikelihood` and
  `periodic-branch-repair-unlikelihood` direct-answer modes
- added `--direct-answer-branch-position`
- selected evidence trains the first answer content character at position `1`
  after the admitted leading answer space
- checkpoint:
  `runs/transformer-answer-v0.40-branch-context32/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- transformer trained for `80` target-loss answer-lesson steps from random
  initialization
- direct answer phase trained for `1000` steps on `9144` weighted direct-answer
  examples
- direct answer mode was `branch-repair-unlikelihood`
- negative weight was `1.0`
- positive weight was `1.0`
- branch position was `1`
- context size was `32`
- average transformer answer target NLL moved from `3.5828` to `2.5427`
- direct answer target loss moved from `3.3496` to `2.3935`
- transformer-only eval-scoped candidate accuracy moved `15/219 -> 37/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- current failure mode changed to repeated `"ten"` loops
- v0.40 improves scored target distribution beyond v0.39 while preserving
  candidate discrimination, but greedy emission still shows weak
  prompt-conditioned branching

`runs/self-improve-v0.40/`:

- kept corpus sources unchanged from v0.39 at `12` admitted facts
- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.39/` and passed
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

`runs/transformer-answer-v0.41-branch-contrast-context32/`:

- added `train_step_with_branch_contrast`
- added `direct_answer_branch_context`
- added `train_direct_answer_branch_contrast_unlikelihood`
- added `branch-contrast-unlikelihood` and
  `periodic-branch-contrast-unlikelihood` direct-answer modes
- added `--direct-answer-contrast-weight`
- this run used branch contrast as every direct-answer update
- checkpoint:
  `runs/transformer-answer-v0.41-branch-contrast-context32/transformer_answer.json`
- this run is preserved as rejected evidence
- direct answer target loss regressed from `3.3496` to `4.4673`
- average transformer answer target NLL regressed from `3.5828` to `4.3931`
- transformer-only eval-scoped candidate accuracy regressed `15/219 -> 16/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- failure mode collapsed to repeated `"a"` output
- lesson: prompt contrast is useful as an objective, but full-dose contrast
  overwhelms the target distribution in the current tiny transformer

`runs/transformer-answer-v0.41-branch-repair-contrast50-context32/`:

- added `periodic-branch-repair-contrast-unlikelihood`
- selected evidence keeps branch repair as the base update and injects branch
  contrast every `50` direct-answer steps
- checkpoint:
  `runs/transformer-answer-v0.41-branch-repair-contrast50-context32/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- transformer trained for `80` target-loss answer-lesson steps from random
  initialization
- direct answer phase trained for `1000` steps on `9144` weighted direct-answer
  examples
- direct answer mode was `periodic-branch-repair-contrast-unlikelihood`
- negative weight was `1.0`
- positive weight was `1.0`
- contrast weight was `1.0`
- branch position was `1`
- contrast interval was `50`
- context size was `32`
- average transformer answer target NLL moved from `3.5828` to `2.4734`
- direct answer target loss moved from `3.3496` to `2.3315`
- transformer-only eval-scoped candidate accuracy moved `15/219 -> 37/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- current failure mode changed to repeated `"te"`/`"e"` loops
- v0.41 improves scored target distribution beyond v0.40 while preserving
  candidate discrimination, but branch probabilities remain nearly prompt-
  independent under greedy decoding

`runs/self-improve-v0.41/`:

- kept corpus sources unchanged from v0.40 at `12` admitted facts
- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.40/` and passed
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

`runs/transformer-answer-v0.42-branch-repair-contrast50-dim8-context32/`:

- selected evidence keeps the v0.41 sparse branch-repair/contrast objective and
  widens the from-scratch transformer from embedding/feed-forward dimensions
  `4/8` to `8/16`
- checkpoint:
  `runs/transformer-answer-v0.42-branch-repair-contrast50-dim8-context32/transformer_answer.json`
- uses the corpus-trained `CharTokenizer`; no pretrained tokenizer, pretrained
  weights, or external embeddings
- transformer trained for `80` target-loss answer-lesson steps from random
  initialization
- direct answer phase trained for `1000` steps on `9144` weighted direct-answer
  examples
- direct answer mode was `periodic-branch-repair-contrast-unlikelihood`
- negative weight was `1.0`
- positive weight was `1.0`
- contrast weight was `1.0`
- branch position was `1`
- contrast interval was `50`
- context size was `32`
- embedding dimension was `8`
- feed-forward dimension was `16`
- average transformer answer target NLL moved from `3.5850` to `2.4129`
- direct answer target loss moved from `3.4278` to `2.2708`
- transformer-only eval-scoped candidate accuracy moved `15/219 -> 37/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- current failure mode changed to the short wrong completion `" te."`
- v0.42 improves scored target distribution beyond v0.41 and reduces runaway
  greedy looping, but branch probabilities remain nearly prompt-independent
- operational note: the wider pure-Python scalar-autodiff run is noticeably
  slower, so future capacity increases should be measured against runtime cost

`runs/self-improve-v0.42/`:

- kept corpus sources unchanged from v0.41 at `12` admitted facts
- standard self-improvement cycle passed on archived `attempt-001`
- forgetting audit compared against `runs/self-improve-v0.41/` and passed
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

`runs/transformer-answer-v0.43-two-layer-toponly-skip-screen-dim8-context32/`:

- added top-layer-only direct-answer updates for stacked transformers so lower
  layers can be treated as fixed feature extractors during bounded screens
- added `--skip-post-direct-snapshot`, which lets a screening run finish and
  save a checkpoint while explicitly recording that the full post-direct
  answer-candidate snapshot was skipped
- checkpoint:
  `runs/transformer-answer-v0.43-two-layer-toponly-skip-screen-dim8-context32/transformer_answer.json`
- transformer trained for `40` target-loss answer-lesson steps from random
  initialization
- direct answer phase trained for `80` top-layer-only steps
- layer count was `2`
- context size was `32`
- embedding dimension was `8`
- feed-forward dimension was `16`
- direct answer mode was `periodic-branch-repair-contrast-unlikelihood`
- direct answer target loss moved from `3.5186` to `3.2436`
- pre-direct answer target NLL moved from `3.5855` to `3.4796`
- pre-direct transformer-only eval-scoped candidate accuracy stayed
  `15/219 -> 15/219`
- raw direct greedy exact answers stayed `0/219 -> 0/219`
- failure mode moved from repeated `"e"` output to repeated `"a"` output
- post-direct candidate snapshot was intentionally skipped, so this is
  training-loop and runtime evidence only, not promotion evidence

`runs/transformer-answer-v0.43-branch-profile-smoke-dim4-context16/`:

- added direct-answer branch profiles to JSONL snapshots
- branch profiles record branch position, branch accuracy, dominant predicted
  tokens, target-token distribution, average target probability, predicted
  probability, target margin, confusion counts, and failed branch examples
- smoke checkpoint:
  `runs/transformer-answer-v0.43-branch-profile-smoke-dim4-context16/transformer_answer.json`
- smoke run trained for `5` target-loss steps and `5` direct-answer branch
  repair steps with the post-direct candidate snapshot skipped
- QA branch-position-1 profile counted `8` records
- QA branch accuracy stayed `1/8`
- dominant QA branch prediction moved from all `"o"` at baseline to all `"y"`
  after five direct updates
- final QA average target margin was negative at about `-0.0048`
- this is self-diagnosis evidence for prompt-independent branch collapse, not
  promotion evidence

`runs/transformer-answer-v0.43-branch-collapse-smoke-dim4-context16/`:

- added `branch-collapse-unlikelihood`, which samples branch examples, finds
  the dominant predicted branch token from QuarkLM's own logits, and uses that
  token as the unlikelihood negative when it differs from the current target
- smoke run trained for `5` target-loss steps and `20` direct-answer steps
- sampled branch pool size came from `--direct-answer-hard-negatives 16`
- post-direct candidate snapshot was intentionally skipped
- QA branch-position-1 profile counted `8` records
- QA branch accuracy regressed `1/8 -> 0/8`
- dominant QA branch prediction moved from all `"o"` to all `"a"`
- average direct-answer target loss regressed `3.5800 -> 3.5827`
- this is rejected repair evidence: dominant-token suppression alone can move
  the collapse without making branches prompt-specific

`runs/transformer-answer-v0.43-periodic-branch-collapse-smoke-dim4-context16/`:

- added `periodic-branch-collapse-unlikelihood`, which applies branch-collapse
  repair every rollout interval and uses ordinary branch repair otherwise
- smoke run trained for `5` target-loss steps and `20` direct-answer steps
- rollout interval was `5`
- sampled branch pool size came from `--direct-answer-hard-negatives 16`
- post-direct candidate snapshot was intentionally skipped
- QA branch-position-1 profile counted `8` records
- QA branch accuracy stayed `1/8 -> 1/8`
- dominant QA branch prediction moved from all `"o"` to all `"n"`
- average direct-answer target loss improved `3.5800 -> 3.5157`
- this is also rejected repair evidence: sparse dominant-token suppression can
  improve loss, but it still leaves a global branch token

`runs/transformer-answer-v0.43-branch-batch-smoke-dim4-context16/`:

- added `branch-batch-contrast-unlikelihood`, which builds a small batch of
  branch contexts with distinct target tokens and trains them in one gradient
  update
- branch batch size was `4`
- smoke run trained for `5` target-loss steps and `20` direct-answer steps
- post-direct candidate snapshot was intentionally skipped
- QA branch-position-1 profile counted `8` records
- QA branch accuracy stayed `1/8 -> 1/8`
- dominant QA branch prediction moved from all `"o"` to all `"y"`
- average direct-answer target loss improved slightly `3.5800 -> 3.5749`
- this is rejected repair evidence: distinct-target branch batching did not
  create prompt-specific branch choices

`runs/transformer-answer-v0.43-periodic-branch-batch-smoke-dim4-context16/`:

- added `periodic-branch-batch-contrast-unlikelihood`, which applies the
  branch-batch objective every rollout interval and uses ordinary branch repair
  otherwise
- branch batch size was `4`
- rollout interval was `5`
- smoke run trained for `5` target-loss steps and `20` direct-answer steps
- post-direct candidate snapshot was intentionally skipped
- QA branch-position-1 profile counted `8` records
- QA branch accuracy regressed `1/8 -> 0/8`
- dominant QA branch prediction moved from all `"o"` to all `"a"`
- average direct-answer target loss improved `3.5800 -> 3.5248`
- this is also rejected repair evidence: the objective improves loss while the
  first answer branch remains global instead of prompt-conditioned

`runs/transformer-answer-v0.43-context-mean-branch-batch-smoke-dim4-context16/`:

- added `--use-context-mean`, an optional representation-side transformer flag
  that adds the mean-pooled prompt context to the final hidden state before the
  feed-forward and language-model head path
- the option is available on both `train` and `answer-train`, is saved in
  `TransformerConfig`, and is recorded in run metrics
- branch batch size was `4`
- rollout interval was `5`
- smoke run trained for `5` target-loss steps and `20` direct-answer steps
- post-direct candidate snapshot was intentionally skipped
- QA branch-position-1 profile counted `8` records
- QA branch accuracy regressed `1/8 -> 0/8`
- dominant QA branch prediction moved from all `"o"` to all `"a"`
- average direct-answer target loss improved `3.5805 -> 3.5252`
- this is rejected representation evidence: adding prompt-average context can
  lower loss without making first answer branches prompt-specific

`runs/transformer-answer-v0.43-context-mean-branch-repair-smoke-dim4-context16/`:

- tested the same `--use-context-mean` representation with sparse branch repair
  instead of branch-batch contrast
- rollout interval was `5`
- smoke run trained for `5` target-loss steps and `20` direct-answer steps
- post-direct candidate snapshot was intentionally skipped
- QA branch-position-1 profile counted `8` records
- QA branch accuracy regressed `1/8 -> 0/8`
- dominant QA branch prediction moved from all `"o"` to all `"a"`
- average direct-answer target loss improved `3.5805 -> 3.5310`
- this is also rejected representation evidence: prompt averaging is too weak
  by itself and the next repair must improve prompt-specific branch separation

`runs/transformer-answer-v0.43-context-projection-branch-repair-smoke-dim4-context16/`:

- added `--use-context-projection`, an optional representation-side transformer
  flag that adds a zero-initialized trainable projection of the mean-pooled
  context to the final hidden state
- the option is available on both `train` and `answer-train`, is saved in
  `TransformerConfig`, stores `context_projection_w` and
  `context_projection_b` in checkpoints, and is recorded in run metrics
- zero initialization preserves baseline logits before training, so any
  projection effect must be learned from the admitted corpus-derived loss
- rollout interval was `5`
- smoke run trained for `5` target-loss steps and `20` direct-answer steps
- post-direct candidate snapshot was intentionally skipped
- all `20` projection parameters moved during the screen; the largest absolute
  projection parameter was about `0.0419`
- QA branch-position-1 profile counted `8` records
- QA branch accuracy regressed `1/8 -> 0/8`
- dominant QA branch prediction moved from all `"o"` to all `"a"`
- average direct-answer target loss improved `3.5802 -> 3.5217`
- this is rejected representation evidence: the learned projection trains and
  lowers loss, but still fails to create prompt-specific first branches

`runs/transformer-answer-v0.43-context-projection-branch-batch-smoke-dim4-context16/`:

- tested the same `--use-context-projection` representation with sparse
  branch-batch contrast instead of branch repair
- branch batch size was `4`
- rollout interval was `5`
- smoke run trained for `5` target-loss steps and `20` direct-answer steps
- post-direct candidate snapshot was intentionally skipped
- all `20` projection parameters moved during the screen; the largest absolute
  projection parameter was about `0.0998`
- QA branch-position-1 profile counted `8` records
- QA branch accuracy regressed `1/8 -> 0/8`
- dominant QA branch prediction moved from all `"o"` to all `"a"`
- average direct-answer target loss improved `3.5802 -> 3.5252`
- this is also rejected representation evidence: distinct branch targets plus
  a learned context projection still collapse to one global branch token

`runs/transformer-answer-v0.43-prompt-attention-branch-repair-smoke-dim4-context16/`:

- added `--use-prompt-attention-summary`, an optional representation-side
  transformer flag that learns an attention-pooled context summary and feeds it
  through a zero-initialized output projection before the final feed-forward
  path
- the option is available on both `train` and `answer-train`, is saved in
  `TransformerConfig`, stores `prompt_summary_query`, `prompt_summary_w`, and
  `prompt_summary_b` in checkpoints, and is recorded in run metrics
- zero output-projection initialization preserves baseline logits before
  training, so the prompt-attention summary must earn influence through
  admitted corpus-derived loss
- rollout interval was `5`
- smoke run trained for `5` target-loss steps and `20` direct-answer steps
- post-direct candidate snapshot was intentionally skipped
- all `20` zero-initialized output projection parameters moved during the
  screen; the largest absolute output projection parameter was about `0.0419`
- QA branch-position-1 profile counted `8` records
- QA branch accuracy regressed `1/8 -> 0/8`
- dominant QA branch prediction moved from all `"o"` to all `"a"`
- average direct-answer target loss improved `3.5802 -> 3.5217`
- this is rejected representation evidence: trainable prompt attention moves
  and lowers loss, but still fails to create prompt-specific first branches

`runs/transformer-answer-v0.43-prompt-attention-branch-batch-smoke-dim4-context16/`:

- tested the same `--use-prompt-attention-summary` representation with sparse
  branch-batch contrast instead of branch repair
- branch batch size was `4`
- rollout interval was `5`
- smoke run trained for `5` target-loss steps and `20` direct-answer steps
- post-direct candidate snapshot was intentionally skipped
- all `20` zero-initialized output projection parameters moved during the
  screen; the largest absolute output projection parameter was about `0.0998`
- QA branch-position-1 profile counted `8` records
- QA branch accuracy regressed `1/8 -> 0/8`
- dominant QA branch prediction moved from all `"o"` to all `"a"`
- average direct-answer target loss improved `3.5802 -> 3.5252`
- this is also rejected representation evidence: branch batching plus
  trainable prompt attention still collapses to one global first answer token

`runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context16/`:

- added direct-answer `branch_context_coverage` diagnostics to every direct
  snapshot
- the diagnostic records branch position, context size, visible context text,
  semantic coverage, missing semantic features, context collisions, ambiguous
  branch contexts, target-token counts, and examples
- smoke run trained for `5` target-loss steps and `5` direct-answer branch
  repair steps with the post-direct candidate snapshot skipped
- QA branch-position-1 profile counted `8` records
- QA branch contexts had `0/8` semantic coverage
- QA branch contexts had only `4` unique context windows for `8` records
- QA branch contexts had `4` ambiguous windows, each with two different first
  target tokens sharing the same visible context
- first ambiguity: `"s ball?\nanswer: "` mapped both place target token `"u"`
  and color target token `"r"`
- this is diagnostic evidence that context-16 branch objectives can be
  underdetermined by the visible branch context

`runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context32/`:

- smoke run trained for `5` target-loss steps and `5` direct-answer branch
  repair steps with the post-direct candidate snapshot skipped
- QA branch-position-1 profile counted `8` records
- QA branch contexts had `8` unique context windows and `0` ambiguous windows
- QA branch contexts still had `0/8` semantic coverage because prompt prefixes
  such as `"question:"` were truncated
- this is diagnostic evidence that context-32 removes literal QA branch
  ambiguity but still lacks complete prompt semantics at the branch point

`runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context80/`:

- tiny smoke run trained for `1` target-loss step and `1` direct-answer branch
  repair step with the post-direct candidate snapshot skipped
- QA branch contexts had `8/8` semantic coverage, `8` unique context windows,
  and `0` ambiguous windows
- all current eval sets reached `219/219` semantic branch-context coverage
  with `0` ambiguous branch contexts
- direct greedy exact remained `0/219 -> 0/219`; direct target loss barely
  moved `3.5846 -> 3.5842` because this was a one-step diagnostic screen
- this is diagnostic evidence that efficient longer-context branch repair is
  the next structured transformer target; another context-16 branch objective
  alone is unlikely to solve prompt-specific branching

`runs/transformer-answer-v0.43-branch-context-gate-smoke-dim4-context16/`:

- added `--direct-answer-require-branch-context-gate`, an opt-in direct-answer
  training guardrail
- the gate passes only when direct-answer branch contexts have complete
  semantic coverage, zero ambiguous target-token contexts, and zero skipped
  branch records
- the gate summary is recorded in every direct-answer JSONL snapshot as
  `branch_context_gate` and in final metrics as
  `direct_answer_branch_context_gate`
- smoke run requested `5` direct-answer branch repair steps at context size
  `16`
- baseline gate failed with `219` missing semantic branch records, `40`
  ambiguous contexts, and `42` collision contexts
- direct-answer training was skipped with reason `branch_context_gate_failed`
- final metrics recorded `steps: 5`, `actual_steps: 0`, and
  `direct_answer_training_skipped: true`
- this is training-loop guardrail evidence: underdetermined context-16 branch
  screens should not be treated as direct-answer improvement evidence

`runs/transformer-answer-v0.43-branch-context-gate-smoke-dim4-context80/`:

- tested the same required branch-context gate at context size `80`
- smoke run requested `1` direct-answer branch repair step
- baseline gate passed with `219/219` semantic branch coverage, `0` ambiguous
  contexts, `0` collision contexts, and `0` skipped records
- final metrics recorded `steps: 1`, `actual_steps: 1`, and
  `direct_answer_training_skipped: false`
- this confirms the gate distinguishes unsafe truncated branch contexts from
  complete branch contexts before training

`runs/transformer-answer-v0.43-branch-context-gated-branchonly-smoke-dim4-context80/`:

- added `--direct-answer-snapshot-mode branch-only` for bounded longer-context
  direct-answer screens
- branch-only snapshots skip greedy completion evals while retaining branch
  profiles, branch-context coverage, and branch-context gate evidence
- required context-80 gate passed with `219/219` semantic branch coverage,
  `0` ambiguous contexts, `0` collision contexts, and `0` skipped records
- smoke run requested `5` direct-answer branch repair steps and recorded
  `actual_steps: 5`
- final metrics recorded `direct_answer_evals_skipped: true`
- this is screening efficiency evidence, not promoted model quality evidence;
  any promotion candidate still needs a full direct-answer snapshot with greedy
  completion evals

`runs/transformer-answer-v0.43-branchonly-periodic-repair-contrast50-dim8-context80/`:

- tested branch-only snapshots with the best prior sparse repair/contrast
  policy at context size `80` and embedding/feed-forward dimensions `8/16`
- required branch-context gate passed with `219/219` semantic branch coverage
  and no ambiguous contexts
- screen requested `100` direct-answer steps and recorded `actual_steps: 100`
- interval train loss moved `6.7890 -> 6.4326`
- QA branch prediction collapsed from all space at baseline to all `"a"` at the
  final snapshot, with final QA branch accuracy `0/8`
- rejected as screening evidence: loss improved, but prompt-specific branch
  separation did not

`runs/transformer-answer-v0.43-branchonly-branch-batch-dim8-context80/`:

- tested branch-batch contrast under the same complete context-80 branch gate
- screen requested `50` direct-answer steps and recorded `actual_steps: 50`
- interval train loss moved `3.4614 -> 3.1976`
- QA branch prediction still collapsed to all `"a"` at the final snapshot, with
  final QA branch accuracy `0/8`
- rejected as screening evidence: branch-batch lowers loss more cheaply, but
  still does not create prompt-specific branch choices

`runs/transformer-answer-v0.43-branch-diversity-target-smoke-dim4-context80/`:

- added structured branch diversity to direct-answer branch profiles:
  `target_unique`, `predicted_unique`, target-token coverage, dominant
  predicted token/rate, collapse status, and missing target tokens
- added `branch_diversity_target` to direct-answer snapshots so multi-target
  branch collapse is a machine-readable target failure
- required context-80 branch-context gate passed with `219/219` semantic branch
  coverage and no ambiguous contexts
- screen requested `5` direct-answer steps and recorded `actual_steps: 5`
- final branch-diversity target failed across all `9` multi-target eval
  profiles
- final QA diversity recorded `target_unique: 8`, `predicted_unique: 1`,
  dominant predicted token `"r"` at rate `1.0`, and target-token coverage
  `0.125`
- this is target-definition evidence: future transformer screens should not
  graduate to full greedy-eval promotion snapshots until this target improves

`runs/transformer-answer-v0.43-branch-diversity-train-smoke-dim4-context80/`:

- added `branch-diversity-unlikelihood`, a direct-answer training mode that
  batches distinct branch targets, trains their targets, suppresses each branch
  context's current wrong prediction, and keeps branch-target contrast
- unit coverage verifies that the objective suppresses a global wrong token
  while raising branch target probability on a small branch batch
- required context-80 branch-context gate passed with `219/219` semantic branch
  coverage and no ambiguous contexts
- screen requested `10` direct-answer steps and recorded `actual_steps: 10`
- final branch-diversity target still failed across all `9` multi-target eval
  profiles
- final QA diversity moved from all `"x"` predictions to all `"b"` predictions,
  target-token coverage improved `0.0 -> 0.125`, and `predicted_unique`
  remained `1/8`
- rejected as training-mode evidence: the objective moves the collapsed token
  but does not yet create prompt-specific branch diversity

`runs/transformer-answer-v0.43-branch-diversity-freezebias-smoke-dim4-context80/`:

- added `--direct-answer-freeze-output-bias`, which removes the transformer's
  output bias from direct-answer updates
- unit coverage verifies that branch-diversity training can leave `bout`
  unchanged while still updating output weights
- required context-80 branch-context gate passed with `219/219` semantic branch
  coverage and no ambiguous contexts
- screen requested `50` direct-answer steps and recorded `actual_steps: 50`
- interval train loss moved `3.6149 -> 3.5016`
- final branch-diversity target still failed across all `9` multi-target eval
  profiles
- final QA diversity moved from all `"x"` predictions to all `"w"` predictions,
  final target-token coverage was `0.0`, and `predicted_unique` remained `1/8`
- rejected as stabilizer evidence: freezing global output bias prevents one
  cheap escape hatch, but the current training path still collapses through
  prompt-independent weights

`runs/transformer-answer-v0.43-branch-target-softmax-freezebias-smoke-dim4-context80/`:

- added `branch-target-softmax-unlikelihood`, a direct-answer mode that applies
  a restricted softmax over the distinct branch targets in each batch
- unit coverage verifies that the objective improves restricted target
  probability on a small branch batch
- required context-80 branch-context gate passed with `219/219` semantic branch
  coverage and no ambiguous contexts
- output bias was frozen with `--direct-answer-freeze-output-bias`
- screen requested `50` direct-answer steps and recorded `actual_steps: 50`
- composite train loss moved `5.6671 -> 5.5820`
- final branch-diversity target still failed across all `9` multi-target eval
  profiles
- QA briefly reached `predicted_unique: 2` at step `20`, then returned to
  `predicted_unique: 1` by step `50` with all `"w"` predictions and final
  target-token coverage `0.0`
- rejected as target-set evidence: direct competition among branch targets can
  transiently disrupt collapse, but does not yet stabilize prompt-specific
  branch diversity

`runs/transformer-answer-v0.43-branch-target-softmax-restorebest-smoke-dim4-context80/`:

- added `--direct-answer-restore-best-branch-snapshot`, which scores
  direct-answer branch snapshots and restores the best measured checkpoint
  before final metrics and checkpoint writing
- unit coverage verifies that the branch-diversity snapshot score prefers a
  less-collapsed prediction profile
- required context-80 branch-context gate passed with `219/219` semantic branch
  coverage and no ambiguous contexts
- output bias was frozen with `--direct-answer-freeze-output-bias`
- screen requested `50` direct-answer steps and recorded `actual_steps: 50`
- final checkpoint was restored from step `40`, with score
  `[0.0, 0.0, -9.0, 0.0, 0.0946, 0.1409, 0.0]`
- final branch-diversity target still failed across all `9` multi-target eval
  profiles
- final QA moved from the prior all-`"w"` target-softmax endpoint to all `"u"`;
  target-token coverage improved `0.0 -> 0.125`, but `predicted_unique`
  remained `1/8`
- rejected as guardrail evidence: best-snapshot restoration preserves the best
  measured branch state but does not yet create prompt-specific branch choices

`runs/transformer-answer-v0.43-prompt-prefix-target-softmax-restorebest-smoke-dim4-context80/`:

- added `--use-prompt-prefix-projection`, a zero-initialized trainable
  projection over non-padding prompt-prefix positions before the final answer
  token
- unit coverage verifies that the projection starts baseline-equivalent, moves
  under training, and round-trips through checkpoint serialization
- required context-80 branch-context gate passed with `219/219` semantic branch
  coverage and no ambiguous contexts
- output bias was frozen and best branch snapshot restoration was enabled
- screen requested `50` direct-answer steps and recorded `actual_steps: 50`
- all `20` prompt-prefix projection parameters moved; max absolute parameter
  value was about `0.0942`
- composite train loss moved `5.6649 -> 5.5679`
- final checkpoint restored from step `40`
- final branch-diversity target still failed across all `9` multi-target eval
  profiles
- final QA stayed collapsed to all `"u"` with target-token coverage `0.125`
  and `predicted_unique` still `1/8`
- rejected as representation evidence: targeted prompt-prefix access is active
  but still insufficient for prompt-specific branch separation

`runs/transformer-answer-v0.43-prompt-position-target-softmax-restorebest-smoke-dim4-context80/`:

- added `--use-prompt-position-projection`, a zero-initialized trainable
  position-specific projection over non-padding prompt-prefix positions before
  the final answer token
- unit coverage verifies that the projection starts baseline-equivalent, moves
  under training, and round-trips through checkpoint serialization
- required context-80 branch-context gate passed with `219/219` semantic branch
  coverage and no ambiguous contexts
- output bias was frozen and best branch snapshot restoration was enabled
- screen requested `50` direct-answer steps and recorded `actual_steps: 50`
- `1108/1284` prompt-position projection parameters moved; max absolute
  parameter value was about `0.0942`
- composite train loss moved `5.6649 -> 5.5679`
- final checkpoint restored from step `40`
- final branch-diversity target still failed across all `9` multi-target eval
  profiles
- final QA stayed collapsed to all `"u"` with target-token coverage `0.125`
  and `predicted_unique` still `1/8`
- rejected as representation evidence: position-specific prompt access is
  active but still insufficient for prompt-specific branch separation

`runs/transformer-answer-v0.43-branch-target-margin-prompt-position-smoke-dim4-context80/`:

- added `branch-target-margin-unlikelihood`, a smooth pairwise target-margin
  loss over each batch's distinct branch targets
- unit coverage verifies that the objective improves a restricted branch logit
  gap on a tiny prompt batch
- required context-80 branch-context gate passed with `219/219` semantic branch
  coverage and no ambiguous contexts
- output bias was frozen, prompt-position projection was enabled, and best
  branch snapshot restoration was enabled
- screen requested `50` direct-answer steps and recorded `actual_steps: 50`
- `1108/1284` prompt-position projection parameters moved; max absolute
  parameter value was about `0.1096`
- train loss moved `4.8973 -> 4.7784`
- final checkpoint restored from step `40`
- final branch-diversity target still failed across all `9` multi-target eval
  profiles
- final QA stayed collapsed to all `"u"` with target-token coverage `0.125`
  and `predicted_unique` still `1/8`
- rejected as target-margin evidence: pairwise target separation lowers the
  bounded loss but still does not stabilize prompt-specific branch choices

`runs/transformer-answer-v0.43-branch-representation-contrast50-prompt-position-smoke-dim4-context80/`:

- added `branch_representation_profiles` to direct-answer snapshots so each
  run records hidden-state pairwise distances before the output head
- added `branch-representation-contrast-unlikelihood`, which penalizes nearly
  identical hidden states for branch contexts with different target tokens
- unit coverage verifies that `final_hidden(context)` matches the output
  logits, the representation profile reports hidden distances, and the
  representation-contrast objective increases hidden distance on a tiny prompt
  batch
- required context-80 branch-context gate passed with `219/219` semantic branch
  coverage and no ambiguous contexts
- output bias was frozen, prompt-position projection was enabled, best branch
  snapshot restoration was enabled, and `--direct-answer-contrast-weight 50.0`
  was used to test whether representation contrast was underweighted
- screen requested `50` direct-answer steps and recorded `actual_steps: 50`
- train loss moved `53.5827 -> 53.4342`; the scale is dominated by the
  high-weight representation term
- final checkpoint restored from step `40`
- final QA stayed collapsed to all `"u"` with target-token coverage `0.125`
  and `predicted_unique` still `1/8`
- final QA different-target hidden distance averaged about `0.00107`, only a
  slight movement from the baseline `0.00097`
- rejected as representation-contrast evidence: the model still has nearly
  indistinguishable hidden states at the answer branch

`runs/transformer-answer-v0.43-branch-representation-contrast50-prompt-position-smoke-dim8-context80-steps40/`:

- tested whether the high-weight representation-contrast path was too narrow at
  embedding/feed-forward dimensions `4/8`
- the matching 50-step dim-8 screen reached step `40` but was too slow for the
  regular loop, so this completed `40`-step run is the evidence artifact
- used embedding/feed-forward dimensions `8/16`, prompt-position projection,
  frozen output bias, best branch snapshot restoration, and
  `--direct-answer-contrast-weight 50.0`
- required context-80 branch-context gate passed with `219/219` semantic branch
  coverage and no ambiguous contexts
- screen requested `40` direct-answer steps and recorded `actual_steps: 40`
- final checkpoint restored from step `10`
- final branch-diversity target still failed across all `9` multi-target eval
  profiles
- final QA stayed collapsed to all `"u"` with target-token coverage `0.125`
  and `predicted_unique` still `1/8`
- final QA different-target hidden distance averaged about `0.00209`, up from
  the dim-4 representation-contrast screen's `0.00107`, but still insufficient
  for branch diversity
- rejected as capacity evidence: more hidden width improves measured separation
  without creating prompt-specific branch choices

The next improvement target is strengthening prompt-conditioned representation
and branch diversity so the direct transformer emits target-specific answers
instead of collapsing to a single global branch token or the short global wrong
answer `" te."`, while preserving the `37/219` candidate-discrimination gain
and v0.42 target-loss gains; then continuing admitted-memory batches,
completing the Python package/import migration to QuarkLM naming, turning more
of the deterministic self-diagnosis and repair policy into
admitted-corpus-trained behavior, and folding the decoder's reliability back
into the broader free-form language model.
