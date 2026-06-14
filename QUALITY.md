# Engineering Quality

Last reviewed for QuarkLM v0.42 on 2026-06-14.

This project should improve its codebase with the same discipline it applies to
model behavior. A change is not promoted only because it works once; it should
be understandable, testable, auditable, and compatible with the closed-world
training boundary.

## SOLID In Python

- Single responsibility: modules should have one clear reason to change.
  Curriculum generation, admission, provenance, training, evaluation, and
  response logic should stay separate.
- Open closed: new corpus checks, eval sets, and report sections should be
  added through focused functions or modules without rewriting stable training
  code.
- Liskov substitution: small data structures and model interfaces should keep
  predictable behavior when loaded from checkpoints or used in tests.
- Interface segregation: command modules should expose narrow helpers that tests
  can call without running full training cycles.
- Dependency inversion: high-level self-improvement orchestration should depend
  on small, explicit functions for training, provenance, audits, and evaluation
  rather than embedding all logic inline.

## Test And Promotion Gates

- Every behavior-level improvement needs a focused unit test.
- Every training improvement needs a recorded baseline and final metrics.
- Failed learned-component evals should preserve failed record details in the
  report so the next cycle can improve from evidence.
- Every new eval set must be included in the responder, answer model, decoder,
  and self-improvement report.
- New admitted memories must be checked by admission probes and a forgetting
  audit against a prior promoted report.
- Direct and paraphrase admission probes must be generated from
  `corpus/admissions.jsonl` and pass the admission-probe audit before a run is
  promoted.
- Glossary probes must be generated from `corpus/glossary.json` probe words and
  pass the glossary-probe audit before a run is promoted.
- Protected held-out prompt leakage must stay at zero.
- The exact eval audit and promotion gate must pass before a run is promoted;
  the self-improvement command should return failure when the gate fails.
- Self-diagnosis should be derived from report artifacts and must declare
  whether it used an external model. The current rule-based diagnosis path must
  remain deterministic and covered by tests.
- Failed promotion attempts should remain evidence. Repairs should target the
  failed report's diagnosis, then rerun the same gate instead of weakening it.
- Self-improvement runs should archive each attempt under
  `attempts/attempt-###/` before updating the top-level latest report.
- Architecture prototypes, including the transformer path, need recorded
  baseline and final metrics before they can be discussed as improvements.
- Selector-assisted architecture metrics must stay separate from free-form
  generation and transformer-only NLL metrics so current evidence is not
  overstated.
- Direct-transformer greedy metrics must stay separate from auxiliary
  generator metrics, especially when the generator is exact but raw transformer
  completions are still failing.
- Self-generated negative training signals, such as first-error unlikelihood,
  must record that the negative came from the model's own prediction rather
  than from an external model or extra corpus.
- Generated-prefix training signals, such as rollout and staged unlikelihood,
  must keep exact greedy completion, candidate discrimination, and failure
  pattern metrics separate because improving one can regress another.
- Periodic rollout schedules must record their interval so candidate
  discrimination and generated-prefix repair dosage can be compared across
  runs.
- Early-stop repair schedules must record their interval and negative weight so
  premature terminator fixes can be compared against repeated-output
  regressions.
- Repeat-loop repair schedules must record their interval, negative weight, and
  final greedy failure pattern because reducing one loop can reveal another.
- Balanced direct objectives should pair self-generated negative repair with
  explicit positive continuation pressure before they are treated as a
  candidate solution to greedy decoding failures.
- Balanced direct runs must record positive weight, negative weight, interval,
  exact greedy output, and scored target metrics because stronger positive
  pressure can still regress candidate discrimination.
- Sequence-repair direct runs must record whether repairs are teacher-forced or
  generated-prefix-based, plus exact greedy output, candidate discrimination,
  direct loss, and final failure pattern. A lower loss is not enough if greedy
  emission regresses or loops are hidden.
- Branch-repair direct runs must record the branch position, exact greedy
  output, candidate discrimination, direct loss, answer NLL, and final failure
  pattern. Better branch likelihood is not enough if the model still chooses a
  prompt-independent repeated sequence.
- Branch-contrast direct runs must record contrast weight, contrast interval,
  branch position, exact greedy output, candidate discrimination, direct loss,
  answer NLL, and final failure pattern. Full-dose contrast and sparse contrast
  must be compared separately because contrast can improve target likelihood or
  collapse the output distribution depending on dosage.
- Capacity changes must record embedding dimension, feed-forward dimension,
  runtime tradeoffs, exact greedy output, candidate discrimination, direct loss,
  answer NLL, and final failure pattern. Wider random models may improve scored
  likelihood without solving prompt-conditioned greedy answers.
- Tokenizer changes must be trained only from admitted corpus text and must not
  import pretrained vocabularies.
- README, STATUS, GOAL, and QUALITY must be reviewed and updated for every
  promoted version so documentation does not drift from the current state.
- Docusaurus docs and marketing pages must be reviewed with every promoted
  version; any page that references current release evidence,
  eval counts, commands, hosting targets, or product positioning must be updated
  in the same release.
- Naming changes must keep product, repository/package slug, import path, and
  command aliases explicit until a full migration is promoted.

## Code Practices

- Prefer pure functions for corpus transforms, audits, feature extraction, and
  report generation.
- Keep run artifacts under `runs/` and make them reproducible from ledgered
  corpus files.
- Keep generated caches out of the project tree during verification.
- Use standard-library dependencies unless a new dependency clearly advances the
  research goal and is admitted deliberately.
