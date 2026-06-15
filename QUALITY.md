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
- Direct-answer snapshots should also record branch-context coverage: visible
  context text, semantic coverage, context collisions, target-token ambiguity,
  and representative missing/ambiguous records. A branch repair screen with
  ambiguous or semantically incomplete branch contexts is diagnostic evidence,
  not proof that another objective can solve prompt-specific branching.
- Direct-answer branch screens should use the branch-context gate when the goal
  is prompt-specific branch repair. A required gate failure should skip training
  and record `actual_steps: 0` plus a skip reason; a run should not be promoted
  as branch-repair evidence unless complete, unambiguous branch contexts are
  proven or the exception is explicitly documented.
- Branch-only direct-answer snapshots may be used for bounded longer-context
  screening, but they must record the snapshot mode and `evals_skipped: true`.
  Treat them as efficiency evidence only until a follow-up full snapshot run
  records greedy completion evals, branch profiles, and gate evidence together.
  A branch-only screen that lowers loss while predicting one token across a
  multi-target eval set is rejected screening evidence, not model-quality
  progress.
- Direct-answer snapshots must retain `branch_diversity_target` once branch
  profiles are present. A failed diversity target blocks promotion-style
  interpretation until a follow-up screen improves predicted-token coverage
  across multi-target eval profiles.
- Diversity-aware branch modes must be judged by `branch_diversity_target` and
  target-token coverage, not by moving collapse from one dominant token to
  another.
- Direct-answer snapshots should retain `branch_representation_profiles` once
  hidden-state diagnostics are present. A representation objective is not a
  promotion unless hidden separation improves enough to change
  prompt-conditioned branch diversity.
- Representation-capacity screens must record runtime practicality, completed
  direct steps, hidden-distance movement, and branch-diversity status. Wider
  hidden states are diagnostic only unless they produce prompt-specific branch
  choices.
- Direct-answer stabilizers that freeze global parameters, such as output-bias
  freezing, must record the frozen option, prove the excluded parameters stayed
  unchanged in unit coverage, and still be judged by branch diversity rather
  than loss movement alone.
- Restricted branch-target objectives, including target-set softmax and
  pairwise target-margin losses, must record both transient and final branch
  diversity. A temporary increase in `predicted_unique` is useful diagnostic
  evidence, but promotion-style interpretation requires the final
  branch-diversity target to improve.
- Best-snapshot restoration must be explicit and auditable. Runs that restore a
  branch snapshot must record the selection score, winning step, whether
  restoration happened, and final branch-diversity target status; preserving a
  better checkpoint is not a promotion unless the final target improves.
- Branch-collapse repair runs must record the sampled branch pool size, branch
  profile before/after, dominant predicted token, direct loss, exact greedy
  output, and whether lower loss actually improved branch accuracy. Penalizing
  a dominant wrong token is not a promotion unless it produces prompt-specific
  branch choices instead of moving collapse to a different global token.
- Branch-batch contrast runs must record branch batch size, dosage, direct
  loss, branch profile before/after, and whether distinct target branches
  become prompt-specific. A batch objective that lowers loss while preserving
  or worsening a global branch token is rejected repair evidence.
- Target-balanced branch batch runs must record that target-bucket sampling was
  used, whether best-snapshot restoration returned to a trained step or the
  baseline, and final per-profile branch diversity. Proving that rare targets
  entered a batch is diagnostic only; if final profiles collapse to one global
  token, target balancing is rejected as a standalone repair.
- Representation-side transformer options, such as context-mean pooling,
  context projection, prompt-prefix projection, prompt-position projection, or
  prompt-attention summaries, must record the option flag, affected training
  commands, direct loss, branch profile before/after, dominant branch token,
  whether new parameters actually moved, and whether the representation
  produces prompt-specific branch choices. Lower loss from a representation
  change is rejected evidence when branch accuracy regresses or the branch still
  collapses to one global token.
- Prompt-signal scaling options, such as
  `--prompt-position-projection-scale`, must prove that the scale applies only
  to the intended residual path, record the scale value, hidden-distance
  movement, projection-weight movement, branch-diversity status, and restored
  snapshot step. A louder prompt residual is diagnostic only unless it produces
  prompt-specific branch choices at the output.
- Open-source structure audits may inform QuarkLM's model/trainer/tokenizer,
  config, checkpoint, and evaluation organization, but they must keep the
  closed-world boundary explicit. Do not import pretrained weights, pretrained
  tokenizer vocabularies, external embeddings, unledgered datasets, unledgered
  training text, or copied model implementations. Any adopted structure should
  be reimplemented QuarkLM-native, covered by focused tests, and documented in
  the evidence trail before it becomes the basis for another repair run.
- Pre-layer-norm or final-normalization transformer paths must be opt-in until
  evidence justifies promotion. They need config round-trip coverage,
  checkpoint compatibility with older models, scalar/float forward parity,
  focused tests for parameter inclusion, and a bounded branch-diversity screen
  before they can be interpreted as architecture progress. A screen that cracks
  collapse in some profiles is useful structural evidence, but it is not
  promotion evidence while the formal branch-diversity target fails or QA and
  heldout remain fully collapsed.
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
- New release identifiers must use SemVer (Semantic Versioning) with
  `vMAJOR.MINOR.PATCH` tags and matching run paths. The current pre-1.0 line
  advances as `v0.100.0`, `v0.101.0`, `v0.102.0`, and so on; do not use
  `XX.YY.ZZ` placeholders or `v1.00` naming. Historical artifacts keep their
  existing names for provenance.
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
