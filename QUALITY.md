# Engineering Quality

Last reviewed for QuarkLM v0.42 and unpromoted v0.43 transformer screens on
2026-06-14.

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
- Direct-answer snapshots should record branch profiles from the model's own
  logits: branch position, branch accuracy, dominant predicted tokens, target
  token distribution, average target probability, and target-vs-top margin.
  These profiles are diagnostic evidence for prompt-independent collapse and
  should guide repair policies before adding another training mode.
- Branch-collapse repair runs must record the sampled branch pool size, branch
  profile before/after, dominant predicted token, direct loss, exact greedy
  output, and whether lower loss actually improved branch accuracy. Penalizing
  a dominant wrong token is not a promotion unless it produces prompt-specific
  branch choices instead of moving collapse to a different global token.
- Branch-batch contrast runs must record branch batch size, dosage, direct
  loss, branch profile before/after, and whether distinct target branches
  become prompt-specific. A batch objective that lowers loss while preserving
  or worsening a global branch token is rejected repair evidence.
- Representation-side transformer options, such as context-mean pooling or
  context projection, must record the option flag, affected training commands,
  direct loss, branch profile before/after, dominant branch token, whether new
  parameters actually moved, and whether the representation produces
  prompt-specific branch choices. Lower loss from a representation change is
  rejected evidence when branch accuracy regresses or the branch still
  collapses to one global token.
- Branch-span direct runs must record the start position, span, exact greedy
  output, candidate discrimination, direct loss, answer NLL, context coverage,
  and final failure pattern. Sweeping later answer positions is not a promotion
  unless it improves the whole answer path rather than moving the loop to a new
  repeated suffix.
- Branch-contrast direct runs must record contrast weight, contrast interval,
  branch position, exact greedy output, candidate discrimination, direct loss,
  answer NLL, and final failure pattern. Full-dose contrast and sparse contrast
  must be compared separately because contrast can improve target likelihood or
  collapse the output distribution depending on dosage.
- Hard-negative branch-contrast runs must additionally record the sampled
  hard-negative count and whether the selected contrast came from the model's
  current branch confusion. A higher candidate count is not sufficient for
  promotion if direct loss, answer NLL, or greedy failure pattern regresses.
- Prompt context-coverage audits must be recorded for transformer answer runs
  when context size changes. Complete semantic-template coverage is necessary
  evidence for longer-context experiments, but it is not a promotion claim
  unless answer metrics also beat the current promoted run.
- Capacity changes must record embedding dimension, feed-forward dimension,
  runtime tradeoffs, exact greedy output, candidate discrimination, direct loss,
  answer NLL, and final failure pattern. Wider random models may improve scored
  likelihood without solving prompt-conditioned greedy answers.
- Depth changes must record layer count, whether intermediate layers require
  full causal state computation, runtime tradeoffs, exact greedy output,
  candidate discrimination, direct loss, answer NLL, and final failure pattern.
  An interrupted run with partial JSONL history is runtime evidence only, not a
  promotion candidate.
- Stacked-transformer screening runs may use top-layer-only direct-answer
  updates or skip an expensive post-direct candidate snapshot only when the
  run metrics explicitly record that choice. A skipped post-direct snapshot is
  acceptable smoke evidence for loop completion and checkpoint writing, but it
  is not promotion evidence without a separate full final candidate evaluation.
- Depth runtime optimizations must prove logit equivalence against the
  unoptimized full-stack path when they change which layer positions are
  computed. Passing equivalence tests does not make a run promotable without a
  complete final metrics artifact.
- Normalization changes must record whether layer normalization was enabled,
  the epsilon value, context coverage, exact greedy output, candidate
  discrimination, direct loss, answer NLL, and final failure pattern. A more
  stable architecture option is not promotable if it merely changes the loop
  shape while regressing scored answer metrics.
- Transformer runtime improvements must be behavior-preserving or covered by
  tests, and their measured effect should be documented when they make longer
  self-improvement runs feasible.
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
