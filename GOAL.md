# Goal Framework

## Objective

Continually improve QuarkLM, a closed-world learner that starts from no
pretrained weights or tokenizer, expands through an admitted model corpus, and
becomes reliable at responding to that corpus.

North star: build toward the world's first language model trained exclusively on
its own admitted dataset. QuarkLM is the human-facing product name, with
`quark-lm` as the repository/package slug. This is an aspiration, not a
completed claim.

Self-improvement applies to every part of the system: model weights, training
code, curriculum design, corpus quality, eval coverage, response reliability,
provenance, documentation, and this goal framework.

## Current Goal Phase

Active phase: experimental PyTorch backend parity for the scalar transformer
reference.

- Preserve scalar Python as the auditable reference implementation.
- Record backend policy, purity metadata, and parity status in transformer
  recipes, metrics, and constraint-first evidence.
- Extend optional PyTorch candidate artifacts only behind deterministic scalar
  parity fixtures; PyTorch runs still cannot count as model-quality evidence
  until profile, optimizer, and training gates pass.
- Keep optimizer-step control evidence separate from numerical update evidence:
  matched cadence, schedule, and update calls do not prove final loss or weight
  parity.
- Keep gradient-clipping evidence scoped: clipping current `tensor.grad` values
  does not prove accumulated-gradient parity, AdamW numerical updates, or final
  loss parity.
- Keep parameter-mutation evidence scoped: observed trainable-tensor changes do
  not prove scalar-equivalent AdamW math, final logits, or final loss parity.
- Compare post-step parameter signatures against scalar training fixtures; a
  mismatch is expected evidence until numerical optimizer parity is implemented.
- Compare actual post-step signatures against scalar-expected AdamW updates from
  current clipped gradients before claiming optimizer math parity.
- Treat matched current-gradient AdamW comparisons as local update evidence, not
  accumulated-gradient, final-logit, or final-loss parity.
- Record accumulated-gradient scope separately: current `tensor.grad` evidence
  must not be treated as replayed backward-pass parity across microsteps.
- Preserve scalar accumulation semantics: QuarkLM applies AdamW to the mean of
  clipped microstep gradients, so PyTorch parity needs a clipped-gradient buffer
  when gradient clipping is active across accumulated microsteps.
- Keep PyTorch accumulation readiness machine-checkable: missing replayed
  backward passes, loss scaling, mean reduction, and clipped-gradient buffers
  must be recorded as pending requirements before training parity can advance.
- Carry a PyTorch accumulation replay plan in candidate artifacts; the plan is
  a microstep recipe, not evidence that replayed backward passes have run.
- Keep replay-control evidence scoped: recorded PyTorch microstep backward
  control does not prove buffered-gradient math, optimizer updates, final
  logits, or final loss parity.
- Record scalar gradient-buffer evidence in training parity fixtures so PyTorch
  can compare raw, clipped, buffered, averaged, and applied gradients before
  claiming accumulated-gradient parity.
- Keep replay-gradient comparison scoped: clipped-gradient signature mismatches
  block accumulated-gradient parity until PyTorch gradients match scalar
  evidence.
- Compare replayed gradient buffers against scalar buffer evidence; buffer
  mismatches keep optimizer-update, final-logit, and final-loss parity blocked.
- Require replay-step alignment before buffer parity: every replayed microstep
  must match the scalar step order before gradient buffers can count.
- Gate replayed AdamW update comparison behind buffer parity; matched trainable
  parameter signatures prove only optimizer-update parity, not final logits or
  final loss.
- Gate final replay evaluation behind optimizer-update parity; matched final
  logits and loss prove evaluation parity only.
- Gate checkpoint compatibility behind final replay evaluation; matched
  round-trip checkpoints prove checkpoint parity only, not promoted PyTorch
  training.
- Gate PyTorch candidate status behind aggregate replay parity: runtime
  readiness, initial loss, backward coverage, optimizer control, replay
  gradients, replay buffers, replay updates, final evaluation, and checkpoint
  compatibility must all match before the candidate can move from pending to
  matched; training parity reports must include this gate, and matched still
  does not mean promoted training.
- Require aggregate replay checks to be status-aware: `passed: true` is not
  enough unless the probe also reports the expected matched status.
- Require aggregate replay checks to be proof-flag-aware: each matched replay
  probe must also expose its explicit parity proof flag.
- Require aggregate replay checks to be schema-aware: replay probes must report
  the expected schema version before they can count.
- Require replay-control count consistency: planned, executed, backward,
  matched-gradient, mismatched-gradient, and microstep-record counts must agree.
- Distinguish real PyTorch from test doubles in runtime evidence. Test doubles
  may validate wiring, but they cannot satisfy aggregate replay parity or count
  as model-quality training evidence.
- Run `quark-lm-torch-runtime` or the equivalent module preflight before any
  real PyTorch parity attempt; a passing runtime report permits an attempt, not
  promotion.
- Record PyTorch training parity attempts with an admitted-curriculum scalar
  fixture, optional PyTorch candidate, training parity report, and compact
  attempt summary; blocked runtime evidence is archived instead of promoted.
- Classify each PyTorch training parity attempt's next unsatisfied requirement
  so the loop can distinguish runtime preflight, training readiness, replay
  parity, and final report failures.
- Keep PyTorch candidate artifacts self-contained by embedding the runtime
  report alongside forward or training evidence.
- Require parity reports to verify embedded runtime reports; training parity
  must also require runtime evidence that allows a real PyTorch parity attempt.
  Runtime preflight proves attempt eligibility, not model-quality training
  evidence or backend promotion.
- Keep PyTorch optional: no dependency requirement, no pretrained assets, no
  unledgered data, and no promoted capability claim.
- Provide PyTorch only through an explicit optional package extra; the default
  scalar install remains dependency-free and canonical.
- Use `float64` as the default PyTorch training parity attempt dtype because
  scalar Python fixtures use double-precision floats; lower-precision runs are
  explicit performance experiments, not the default parity gate.
- Keep optional real-runtime PyTorch parity covered by skip-safe tests: default
  scalar environments skip cleanly, while installed PyTorch environments must
  match scalar training replay evidence before any backend promotion claim.
- Embed an explicit PyTorch backend-promotion gate in training parity attempts;
  matched replay parity is fixture evidence only and must not silently become a
  promoted/general training backend. The gate should report exact closed-world
  boundary fields when the boundary fails.
- Validate PyTorch training parity attempt summaries before trust or write so
  attempt status, next requirements, promotion gate, evidence scope,
  closed-world boundary, artifact paths, and artifact payload consistency remain
  machine-checkable, including a report rebuilt from the paired fixture and
  candidate payloads.
- Treat focused backend parity tests, full Python discovery, docs builds, and
  code-quality review as the evidence gate for this phase.

## Current Canonical Evidence

Keep this file as the durable goal contract. Long version history and detailed
run evidence live in the docs and shared state files.

| Evidence | Canonical location |
| --- | --- |
| Current status | `STATUS.md` |
| Current docs/marketing state | `sites/shared/current-state.json` |
| Full evidence trail | `sites/docs/docs/learn/current-evidence.mdx` |
| Research grounding | `sites/docs/docs/learn/research-grounding.md` |
| Branch-diversity research | `sites/docs/docs/learn/branch-diversity-research.md` |
| Implementation map | `sites/docs/docs/learn/research-implementation-map.md` |
| Historical evidence archive | `sites/docs/docs/learn/historical-evidence.md` |
| RC boundary | `RC_SPEC.md`, `RC_GAP_AUDIT.md`, `RC_CHECKLIST.md` |

Current evidence summary:

- Promoted responder evidence: `runs/self-improve-v0.42/`.
- Latest transformer diagnostic screen:
  `runs/transformer-answer-v0.115.0-hidden-projection-margin-candidate-step1-dim4-context80/`.
- v0.115 transformer posture: `10/11` constraints pass, but
  `branch_diversity_target` still blocks neural promotion; retrieval memory is
  exact but does not count as weight learning.
- Current PyTorch backend posture: tiny CPU `float64` real-runtime training
  replay parity matches scalar evidence, but PyTorch remains optional,
  unpromoted, and not a general training backend.
- Next model direction: profile-balanced routing repair with
  representation-separation acceptance checks.

Public surfaces:

- Docusaurus docs target `docs.quark-lm.eidetic-labs.com` and are hosted by
  Read the Docs.
- The standalone marketing page targets `quark-lm.eidetic-labs.com` and is
  hosted by GitHub Pages.
- Deployment details live in `sites/DEPLOYMENT.md`.

## Guardrails

- No pretrained weights.
- No pretrained tokenizer.
- No external embeddings.
- No unledgered training text.
- Scalar Python remains QuarkLM's canonical reference implementation because it
  keeps the model math inspectable and dependency-free. PyTorch is the planned
  performance backend for scalable training, batched evaluation, optimized
  attention, and hardware acceleration. PyTorch is allowed as a runtime
  library; it does not change the closed-world boundary unless pretrained
  weights, pretrained tokenizers, external embeddings, copied model code, or
  unledgered data are introduced. NumPy is not a required interim backend and
  should only be added later for a narrow diagnostic need.
- Any future tokenizer upgrade must be trained only from admitted corpus text;
  pretrained vocabularies are outside the boundary.
- Every run must record an untrained or prior-checkpoint baseline and trained
  metrics.
- Every reliability claim needs an evaluation artifact.
- Improvements must preserve dataset exclusivity: new training examples must be
  generated from, or explicitly added to, the admitted corpus.
- Weight updates are part of the goal, but only as versioned training artifacts:
  no silent in-place mutation, no outside weights, no unmeasured promotion.
- Failed training or promotion attempts must remain archived evidence, not
  overwritten by later repair attempts.
- Retrieval success means the corpus can serve admitted knowledge; it does not
  prove the neural weights learned that knowledge.
- Candidate records, generated lessons, generated probes, diagnosis notes, and
  repair proposals are not training data until admitted into the ledgered corpus
  and converted into curriculum lessons.
- "Self" means operational self-knowledge only: the model may know its corpus
  boundary, weight-update process, admitted dataset, unknown policy, and current
  learning loop. It must not claim consciousness or subjective experience.
- "I learned something new" means the information was admitted into the
  ledgered corpus, converted into training candidates or lessons, used in a
  measured update when appropriate, and preserved as auditable evidence.
- Code quality is part of self-improvement: use SOLID-aligned Python module
  boundaries, focused tests, clear artifacts, and small pure functions for
  corpus transforms, audits, feature extraction, and reporting.
- Public documentation is part of self-improvement: README, Docusaurus docs,
  STATUS, and marketing pages must update with promoted releases whenever they
  describe current product state, commands, evals, evidence, hosting, or roadmap
  commitments.
- Research planning is part of self-improvement: new objective modes should be
  preceded by a stated hypothesis, allowed data boundary, planned artifacts,
  acceptance gates, and promotion decision criteria.
- Eventual self-improvement should not depend on an external model shaping the
  learner. Near-term guidance may use deterministic, auditable rules over
  QuarkLM's own reports; the long-term target is repair proposal and selection
  learned from admitted artifacts and versioned outcomes.

## Current Loop

1. Admit or refine corpus data through ledgered files.
2. Regenerate curriculum, probes, retrieval memory, and provenance artifacts.
3. Build source-backed training candidates from admitted data, retrieval
   evidence, replay plans, and failure reports.
4. Declare experiment intent before training: hypothesis, allowed data,
   planned artifacts, acceptance gates, failure criteria, and promotion
   decision criteria.
5. Run deterministic verifier checks before trusting a training plan.
6. Train from random initialization or from an explicit closed-world checkpoint
   only when the checkpoint is part of the declared experiment.
7. Record baseline metrics, final metrics, checkpoint metadata, recipe
   artifacts, and constraint-first promotion evidence.
8. Evaluate closed-world exactness, unknown policy, retention, prompt leakage,
   target coverage, branch diversity, and docs/current-state alignment.
9. Promote only when constraints pass. Rejecting a run should still produce
   diagnostic evidence for the next repair.
10. Archive attempts and keep failed evidence visible.
11. Update README, STATUS, Docusaurus docs, shared current state, and marketing
    when they reference current behavior, release status, commands, evals, or
    hosting.
12. Use QuarkLM's own reports and deterministic diagnosis surfaces to choose the
    next repair without relying on external model shaping.

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
6. Held-out fact probes are not trained with their exact evaluation prompt
   forms; they may only enter learned responders through admitted fact-style
   lessons.
7. QA-training facts may include question-style and fact-style lessons so the
   model can learn transfer between surface forms without leaking held-out
   prompt answers.
8. Generated bridge lessons may be added from admitted facts when they improve
   transfer, but protected held-out evaluation prompts must remain absent from
   lesson files.
9. A promoted self-improvement run must pass the recorded promotion gate.
10. Experimental architecture checkpoints, such as transformer screens, are
    evidence only for the behavior they actually show. Lower loss is not a
    reliable-answer claim until answer evals and promotion gates pass.
11. A bounded architecture screen may skip expensive evaluation only when the
    run metrics record that skip. Such a run can prove loop completion or
    diagnostic movement, but not promotion-quality model behavior.

## Reliability Strategy

The corpus response model is the reliability rail: it learns a tiny fact table
from admitted `fact:` lines and answers source-grounded questions exactly. The
response model gives the project a grounded teacher and oracle while neural
components learn to produce those answers directly.

The learned answer model is the first learned bridge. It starts with random
softmax weights, trains only on corpus-derived question/answer lessons, and is
evaluated on exact closed-world answers.

The learned answer decoder moves that bridge closer to language modeling by
generating answer characters one by one from prompt-conditioned weights.

The tiny transformer learner is the architecture path toward a more
recognizable language model. It starts from random weights, uses QuarkLM's
corpus-trained character tokenizer, and currently trains with a
dependency-free scalar autodiff engine. The scalar path remains the auditable
reference; the planned PyTorch path is the performance backend once it passes
parity gates. The transformer is not yet the reliable response path and must
mature under the same corpus and eval gates.

The self-diagnosis layer is the current bridge toward autonomous improvement.
It reads QuarkLM's own run reports and recommends the next repair or promotion
action with deterministic rules. Future versions should move that
diagnosis-and-repair policy into admitted, trainable artifacts while preserving
the same evidence boundary.

## Engineering Quality Strategy

The codebase should stay small but intentionally shaped. Keep orchestration,
corpus provenance, admission, curriculum, response, retrieval, learned answer
selection, generative decoding, transformer training, verifier checks, and
promotion decisions behind narrow module boundaries.

Tests should scale with risk:

- focused unit tests for corpus transforms, audits, artifacts, and objective
  mechanics;
- integration tests for self-improvement and transformer run contracts;
- docs/site builds when public surfaces or shared current state change;
- full Python discovery before release-candidate packaging or public upload.

Docs are part of quality. README should stay a concise front door; durable
model philosophy, version evidence, deployment details, and long-form release
history belong in Docusaurus and shared current-state artifacts.
