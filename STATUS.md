# QuarkLM - Status

**Status:** Experimental research scaffold
**Active version:** v0.115.0 hidden-projection margin candidate;
promoted responder evidence remains v0.42
**RC posture:** Research Prototype RC is near; Language Model RC is blocked by
transformer branch routing.
**Last updated:** 2026-06-19
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
- Corpus-only retrieval memory in `src/closed_world_lm/memory_retrieval.py`
  that serves admitted/story/self/learning/glossary knowledge immediately
  without external embeddings or weight updates.
- Memory-guided consolidation planning in
  `src/closed_world_lm/memory_consolidation.py` that ranks retrieval-served
  profiles whose neural branch predictions still fail branch-diversity gates.
- Gated memory-consolidation direct-answer training that consumes a declared
  source `memory_consolidation_plan.json` and records consumed target profiles,
  prioritized attempts, acceptances, and rejections without treating retrieval
  success as promotion.
- Explicit target-profile-to-source-label mapping for expanded consolidation
  windows, including `heldout`, `qa`, `admissions`, and
  `admission_paraphrases`.
- Plan-derived missing first-token memory-consolidation pressure that extracts
  missing first-token target maps from a source plan, trains only under guarded
  coverage-gain checks, and records candidates, attempts, acceptances,
  rejections, fallback acceptances, and rejection reasons.
- Remaining-collapsed memory-consolidation targeting that requires a source
  plan's `collapsed_memory_backed_profiles`, consumes only unresolved collapsed
  profiles, and records the collapsed-target contract in replay and guard
  artifacts.
- Profile-specific missing first-token memory-consolidation targeting that maps
  each admitted source label to only the unresolved target profiles it can
  support before guarded weight updates are evaluated.
- Branch-diversity root-cause diagnostics in
  `src/closed_world_lm/branch_diversity_diagnostics.py`, classifying failed
  profiles as global collapse, profile-local collapse, target-routing gaps,
  target-rank burial, wrong diversity, or mixed gaps before another objective
  is introduced.
- Branch routing audits that attach output-bias escape risk, prompt-to-branch
  representation separation, and profile/target imbalance diagnostics to each
  direct-answer branch snapshot.
- Branch logit-prior profiles that decompose dominant-token wins into output
  bias, hidden projection, or mixed pressure before another repair objective is
  selected.
- Branch hidden-projection margin training that compares target-token
  `hidden * output_weight` contributions directly, so a candidate can repair
  target routing without treating output bias as the primary repair surface.
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
- Public surfaces: Docusaurus docs at `docs.quark-lm.eidetic-labs.com` hosted
  by Read the Docs, and a standalone static marketing page at
  `quark-lm.eidetic-labs.com` hosted by GitHub Pages. The marketing site is not
  Docusaurus.
- Release-candidate planning in `RC_SPEC.md`, `RC_GAP_AUDIT.md`, and
  `RC_CHECKLIST.md`, separating Research Prototype RC readiness from Language
  Model RC readiness.
- SOLID-aligned quality guidance in `QUALITY.md`.
- Paper-grounded research guidance for continual learning, replay,
  self-generated candidate lessons, retrieval rails, model editing boundaries,
  transformer architecture, and tokenizer timing.
- Open-source mechanics audit guidance for trainer boundaries, profile-aware
  replay, checkpoint scoring, tokenizer artifacts, and candidate lesson
  acceptance without copying outside code or importing outside training data.
- Forward research plan for implementing the self-improvement operating system:
  experiment registry, corpus governance, candidate quarantine, deterministic
  verifier checks, replay extraction, training recipes, and constraint-first
  promotion gates before the next objective mode.
- Deep research review that cross-references primary research, official
  open-source implementation references, and the current QuarkLM codebase to
  define the full operating-system stack required before the next major run.
- Experiment intent registry for self-improvement and transformer
  answer-training runs. Runs now declare hypothesis, allowed data sources,
  planned artifacts, recipe id, gates, failure criteria, notes, and a decision
  artifact before they are trusted as evidence.
- Standalone replay planning in `src/closed_world_lm/replay_plan.py`, with
  focused tests for profile-aware deficits, legacy branch records, fallback
  replay records, profile keys, and JSON artifact safety.
- Corpus hygiene and training-plan artifacts in
  `src/closed_world_lm/corpus_hygiene.py`. Self-improvement and transformer
  answer-training runs now write `corpus_hygiene.json` and `training_plan.json`
  with source mixtures, duplicate checks, train/eval prompt overlap, candidate
  ratios, rare-profile coverage, allowed data sources, and planned artifacts.
- Research implementation map in `RESEARCH_IMPLEMENTATION_MAP.md` and
  `sites/docs/docs/learn/research-implementation-map.md`. v0.74 ties research
  clusters and open-source mechanics to QuarkLM implementation gaps and shifts
  candidate quarantine to v0.75 so the next mechanics are source-backed before
  code is added.
- Candidate quarantine artifacts in
  `src/closed_world_lm/candidate_quarantine.py`. Self-improvement and
  transformer answer-training runs now write `candidate_quarantine.json` with
  lifecycle state, manifest counts, transition policy, and an explicit rule
  that candidate records are not training data until admitted into the ledgered
  corpus and converted into curriculum lessons.
- Deterministic closed-world verifier checks in
  `src/closed_world_lm/closed_world_verifier.py`. Self-improvement and
  transformer answer-training runs now write `closed_world_verifier.json`,
  embed verifier summaries in `training_plan.json`, and declare verifier
  approval as a required run-intent gate before training evidence can be
  trusted.
- Training recipe and constraint-first promotion artifacts in
  `src/closed_world_lm/training_recipe.py`. Self-improvement and transformer
  answer-training runs now write `training_recipe.json` and
  `constraint_first_promotion.json`; transformer decisions cannot promote from
  loss, NLL, rank, or top-k evidence unless closed-world constraints pass
  first.
- Transformer responsibility surfaces in
  `src/closed_world_lm/transformer_experiment.py`,
  `src/closed_world_lm/transformer_training.py`, and
  `src/closed_world_lm/transformer_objectives.py`. Answer-training now keeps
  artifact contracts, experiment/recipe decisions, JSONL snapshot writing,
  shuffled training cursors, loss averaging, and the direct-answer objective
  catalog behind narrow modules while preserving the public CLI.
- Transformer model/checkpoint surfaces in
  `src/closed_world_lm/transformer_model.py`. Model, optimizer, and generation
  config dataclasses, validation, checkpoint identity, closed-world dataset
  metadata, arg-to-config adapters, and run metadata now live outside the
  transformer monolith while remaining re-exported for compatibility.
- Transformer eval/checkpoint-load surfaces in
  `src/closed_world_lm/transformer_checkpoint.py` and
  `src/closed_world_lm/transformer_eval.py`. Checkpoint payload validation,
  checkpoint summaries, probe loading, eval candidate collection, generic
  transformer scoring, eval report assembly, samples JSONL writing, and eval
  JSON writing now live outside the transformer monolith.
- Profile-aware direct-answer replay records, per-profile deficit and
  preservation accounting, replay-plan artifacts, and profile-isolation tests
  for transformer repair screens.
- Source probes for known, unknown, held-out, paraphrase, ownership, self,
  learning, admission, admission-paraphrase, and glossary answers.
- Optional PyTorch training parity attempt artifacts via
  `quark-lm-torch-training-parity`. The command builds an admitted-curriculum
  scalar fixture, optional PyTorch candidate, training parity report, and
  compact attempt summary; absent PyTorch records blocked runtime evidence
  instead of a promoted backend claim. The summary names the next unsatisfied
  requirement so the loop can separate runtime, readiness, replay, and report
  failures.
- Optional PyTorch runtime installation is exposed as the `pytorch` package
  extra. It is not part of the default scalar install and does not introduce
  pretrained weights, tokenizers, external embeddings, or unledgered data.
- Local real-runtime evidence now clears the PyTorch training replay parity
  attempt on CPU with `float64`: runtime readiness, initial loss, replay
  gradients, replay buffers, optimizer update, final evaluation, checkpoint
  compatibility, and the training parity report all match scalar evidence.
  A skip-safe optional integration test now records this proof when PyTorch is
  installed and skips under the default scalar environment. This is parity
  evidence only; PyTorch is still not a promoted training backend.
- PyTorch training parity attempts now include an explicit backend-promotion
  gate. It is expected to fail today, preserving the boundary between matched
  fixture replay parity and any future promoted/general PyTorch trainer. The
  gate reports exact closed-world boundary fields when they fail.
- PyTorch training parity attempt summaries are validated before being trusted
  or written, including attempt status, next requirements, promotion-gate,
  closed-world boundary, evidence-scope, artifact-path, and artifact-payload
  consistency checks. The stored training parity report must match a report
  rebuilt from the paired fixture and candidate payloads, and the stored
  backend-promotion gate must match a gate rebuilt from the candidate, report,
  and closed-world boundary. The next-requirements diagnosis must also rebuild
  from the candidate runtime report, candidate, and report. Written summaries
  now carry SHA-256 payload hashes for sibling artifacts, and written attempt
  directories are reloaded through the same validation contract before the
  writer returns. Recorded artifact paths must resolve to the loaded files, and
  the CLI can audit an existing attempt directory with `--verify-existing`
  without rebuilding it. The optional public backend surface exposes the
  written-attempt file map, hash algorithm, hash builder, and loader so
  contributors can inspect the same persisted audit contract without reaching
  through private module paths.

## Research Grounding

QuarkLM's current research posture is documented in
`sites/docs/docs/learn/research-grounding.md`. The 2026-06-14 research pass
maps QuarkLM to self-improvement lifecycle work, continual learning, replay,
synthetic-recursion risk, small-data language learning, data hygiene, retrieval
rails, model editing boundaries, transformer mechanics, and evaluation
contamination. The project is adjacent to those areas, but its stricter claim
is that model weights and tokenizer state should be trained only from admitted,
ledgered data.

The practical near-term guidance is:

- make replay a first-class training primitive;
- keep corpus admission separate from model belief;
- accumulate original admitted records instead of replacing them with
  model-generated summaries;
- add corpus hygiene reports for duplicates, source mixtures, synthetic
  candidate ratios, and rare-record coverage;
- verify self-generated lessons before admission or training;
- promote only through retention, forgetting, leakage, unknown-policy, and
  branch-diversity gates;
- train the transformer from coverage deficits, not only from already-covered
  branch targets;
- use profile-aware replay-plan artifacts to gate full-stack direct-answer
  repair runs and reject any snapshot that improves rank by erasing profile
  coverage or branch diversity;
- defer model editing and self-rewarded grading until locality, side effects,
  and verifier quality are measurable inside the closed world.

The v0.66 mechanics audit is documented in `MECHANICS_AUDIT.md` and
`sites/docs/docs/learn/open-source-mechanics-audit.md`. It compares public LLM,
tokenizer, continual-learning, transparency, and self-improvement mechanics as
design references only. It concludes that QuarkLM should pause global
branch-loss churn and make profile-aware replay plans, per-profile deficits,
per-profile preservation, replay-plan artifacts, and profile-isolation tests
the next implementation gate.

v0.67 implements that gate for the direct-answer transformer path. Branch
replay records can now carry profile keys, replay deficits and represented
target preservation are computed per profile, and profile-aware modes write
`direct_answer_replay_plan.json` before training. The bounded smoke run
`runs/transformer-answer-v0.67-profile-aware-replay-plan-smoke-dim4-context80/`
completed one gated direct step, wrote a replay plan for `9144` branch records
across `21` profiles, and confirmed that `qa:place` and `qa:color` can expose
different coverage floors in the same artifact. This is mechanics-readiness
evidence, not promotion evidence.

v0.68 spends that mechanism on the comparable full-stack direct-answer repair
screen. The run
`runs/transformer-answer-v0.68-fullstack-profile-aware-preserving-deficit-smoke-dim4-context80/`
completed `50/50` direct steps and wrote the same profile-aware replay-plan
shape, but best-snapshot scoring restored step `0`. Training improved rank
evidence at step `40`, yet it collapsed QA and heldout target-token coverage to
`0.125` with predicted diversity `1/8`, so the gate correctly rejected the
trained snapshots.

v0.69 records the forward research plan in `FORWARD_RESEARCH_PLAN.md` and
`sites/docs/docs/learn/forward-research-plan.md`. The plan cross-references
papers and public implementation mechanics from continual learning, replay,
self-generated data, self-feedback, verifier-style rewards, model collapse,
small-data training, transparent open models, data hygiene, and trainer recipe
systems. The implementation decision is to pause further direct-answer objective
churn until QuarkLM has experiment registry, replay extraction, corpus hygiene,
candidate quarantine, closed-world verifier checks, training recipes, and
constraint-first promotion gates.

v0.70 records the deeper cross-referenced review in `DEEP_RESEARCH_REVIEW.md`
and `sites/docs/docs/learn/deep-research-review.md`. It reviews primary papers,
official project mechanics, and QuarkLM's current codebase. The implementation
decision is to treat experiment intent, training recipes, corpus hygiene,
candidate quarantine, deterministic verification, replay extraction, and
constraint-first promotion as required operating-system mechanics before the
next larger transformer screen. The experiment registry is now the v0.71
implementation target.

v0.71 implements that target in `src/closed_world_lm/experiment_registry.py`.
Self-improvement answer cycles and transformer answer-training runs now write
`experiment_intent.json` before training and include the final intent decision
in their reports or metrics. From v0.77 onward, transformer screens close
through the constraint-first promotion report.

v0.72 extracts replay planning into `src/closed_world_lm/replay_plan.py`.
Transformer training still uses the existing profile-aware replay behavior, but
branch replay normalization, profile grouping, coverage-floor summaries, and
replay-plan JSON shape are now standalone mechanics with focused tests.

v0.73 adds `src/closed_world_lm/corpus_hygiene.py` and wires
`corpus_hygiene.json` plus `training_plan.json` into self-improvement and
transformer answer-training runs. These artifacts report source mixtures,
duplicates, train/eval prompt overlap, candidate ratios, rare-profile
coverage, allowed data sources, planned artifacts, and replay-plan summaries
when profile-aware replay is written.

v0.74 adds `RESEARCH_IMPLEMENTATION_MAP.md` and the matching Docusaurus Learn
page. It cross-references transformer, continual-learning, small-data,
self-generated-data, verifier, tokenizer, data-curation, transparent open-model,
and public training-stack sources against QuarkLM's implementation gaps. The
decision is to treat candidate quarantine as v0.75, deterministic verifier
checks as v0.76, recipe and constraint-first promotion as v0.77, and the first
transformer responsibility surfaces as v0.78. v0.79 extracts model/config and
checkpoint metadata surfaces, and v0.80 extracts eval/checkpoint-load surfaces
before another larger repair run.

v0.75 adds `src/closed_world_lm/candidate_quarantine.py`, the Docusaurus
Operate page for candidate quarantine, and `candidate_quarantine.json` artifacts
for self-improvement and transformer answer-training paths. Training plans now
link the quarantine manifest and summarize candidate counts.

v0.76 adds `src/closed_world_lm/closed_world_verifier.py`, the Docusaurus
Operate page for closed-world verifier checks, and `closed_world_verifier.json`
artifacts for self-improvement and transformer answer-training paths. The
verifier approves training plans only when the data boundary is closed,
candidate records are excluded from training, candidate quarantine is valid, and
protected train/eval overlap checks pass.

v0.77 adds `src/closed_world_lm/training_recipe.py`, the Docusaurus Operate
page for training recipes and constraint-first promotion, and
`training_recipe.json` plus `constraint_first_promotion.json` artifacts for
self-improvement and transformer answer-training paths. Recipes bind model,
tokenizer, data, objective, optimizer, artifacts, gates, replay status, and
rerun surfaces. Constraint-first reports block quality metrics until
closed-world constraints pass.

v0.78 adds `src/closed_world_lm/transformer_experiment.py`,
`src/closed_world_lm/transformer_training.py`, and
`src/closed_world_lm/transformer_objectives.py`. Transformer answer-training
keeps the CLI stable while extracting the artifact contract, intent/recipe
decision logic, JSONL history writing, shuffled training cursors, loss
averaging, and direct-answer objective catalog into narrow, separately tested
surfaces.

v0.79 adds `src/closed_world_lm/transformer_model.py`. Model, optimizer, and
generation config dataclasses, validation, checkpoint format identity,
closed-world dataset metadata, arg-to-config adapters, and run metadata now
live behind a model/checkpoint surface outside the transformer monolith while
remaining re-exported from `transformer_char_model.py` for compatibility.

v0.80 adds `src/closed_world_lm/transformer_checkpoint.py` and
`src/closed_world_lm/transformer_eval.py`. Checkpoint payload loading and
identity validation, checkpoint summaries, probe loading, candidate collection,
generic eval scoring, report assembly, samples JSONL writing, and eval JSON
writing now live behind focused surfaces while preserving CLI behavior and
artifact shapes. The next transformer mechanic can return to objective-repair
work with the operating surfaces in place.

v0.81 adds
`branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood`.
The objective reuses profile-aware replay plans, preserving-deficit pressure,
and recipe/promotion surfaces while adding balanced per-profile target-share
loss, so a multi-target profile gets explicit pressure for every replay target
instead of over-preserving a single represented target. This is a focused
objective-mechanics checkpoint; model-quality promotion still requires a
future full-stack screen to pass the closed-world constraints.

v0.82 runs the matching full-stack screen at
`runs/transformer-answer-v0.82-fullstack-profile-target-share-smoke-dim4-context80/`
and fixes the transformer metrics purity report so the constraint-first gate
sees `external_embeddings: false`. The screen wrote experiment intent, corpus
hygiene, training plan, candidate quarantine, verifier, recipe, replay plan,
constraint-first report, metrics, tokenizer, optimizer, lessons, and checkpoint
artifacts. It completed `50/50` direct steps with `7` JSONL rows; the
branch-context gate passed and coverage preservation restored step `0`, but
branch diversity still failed. Step `40` improved QA average target rank to
`9.125` while collapsing QA and heldout to one `"c"` prediction with `0.0`
target-token coverage, so the run is rejected evidence.

v0.83 adds
`branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
The objective keeps the profile target-share, deficit, preservation, replay
plan, and recipe/promotion surfaces, then adds a prompt-specific sibling-target
margin so each replay context is trained to outrank other targets from the same
profile. A focused unit test proves the new margin lifts a context-specific
target more than the v0.82 target-share pressure. The matching screen at
`runs/transformer-answer-v0.83-fullstack-prompt-ownership-smoke-dim4-context80/`
wrote the modern artifacts, passed the verifier, branch-context, and purity
gates, and completed `50/50` direct steps with `7` JSONL rows. Step `50`
improved QA average target rank to `8.625` and heldout rank to `8.5`, but
trained snapshots still collapsed QA and heldout to one `"c"` prediction with
`0.0` target-token coverage. Best-snapshot scoring restored step `0`, so the
run is rejected evidence and the next repair needs coverage-preserving
prompt-specific training.

v0.84 adds
`branch-balanced-context-profile-baseline-anchored-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
The objective anchors replay preservation to the baseline profile-aware replay
predictions captured before direct-answer training starts, so coverage
preservation no longer follows current prediction drift. Focused tests verify
baseline prediction overrides in profiled replay batches and prove anchored
preservation protects a covered target better than the dynamic v0.83 path. The
matching screen at
`runs/transformer-answer-v0.84-fullstack-baseline-anchored-prompt-ownership-smoke-dim4-context80/`
wrote the modern artifacts, recorded `562` active baseline prediction anchors,
passed the verifier, branch-context, and purity gates, and completed `50/50`
direct steps with `7` JSONL rows. Step `40` improved QA average target rank to
`8.0` and heldout rank to `8.375`; QA and heldout still collapsed to one
`"i"` prediction, but target-token coverage held at `0.125` instead of the
v0.83 `0.0` collapse. Best-snapshot scoring restored step `0` because trained
snapshots still regressed below the `0.25` coverage floor, so the run is
rejected evidence.

v0.85 adds
`branch-balanced-context-profile-baseline-floor-gated-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
It keeps the v0.84 baseline replay anchors and adds an update guard that probes
branch-profile target-token coverage after each attempted direct-answer update.
If the update falls below the step-0 profile-wise coverage floor, the loop
restores the prior model and optimizer state before continuing. Focused tests
prove the mode records the replay anchors and guard accounting. The matching
screen at
`runs/transformer-answer-v0.85-fullstack-baseline-floor-gated-prompt-ownership-smoke-dim4-context80/`
wrote the modern artifacts, recorded `562` active baseline prediction anchors,
checked `50/50` attempted direct-answer updates, and rejected all `50` as unsafe.
Every recorded snapshot preserved baseline/final QA and heldout target-token
coverage at `0.25`, predicted diversity at `3/8`, QA average target rank at
`13.25`, and heldout average rank at `13.375`. This is rejected but useful
safety evidence: the guard prevents unsafe forgetting, but no update is accepted
and `branch_diversity_target` still fails across all `9` multi-target profiles.

v0.86 adds
`branch-balanced-context-profile-baseline-floor-adaptive-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
It keeps the v0.85 guard and retries the same direct-answer update at
learning-rate scales `1.0`, `0.25`, `0.05`, and `0.01`, restoring model,
optimizer, and RNG state before each retry. Focused tests prove the mode records
adaptive retry accounting. The matching screen at
`runs/transformer-answer-v0.86-fullstack-baseline-floor-adaptive-prompt-ownership-smoke-dim4-context80/`
wrote the modern artifacts, recorded `562` active baseline prediction anchors,
checked `50/50` steps, attempted `200` scaled updates, and rejected all `200` as
unsafe. Every recorded snapshot preserved baseline/final QA and heldout
target-token coverage at `0.25`, predicted diversity at `3/8`, QA average target
rank at `13.25`, and heldout average rank at `13.375`. This is rejected evidence:
smaller learning-rate scales do not make the update safe, which sets up the
v0.87 repair-retry screen.

v0.87 adds
`branch-balanced-context-profile-baseline-floor-repaired-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
It keeps the v0.86 adaptive guard and adds one bounded baseline-covered anchor
repair before a failed retry is accepted or rejected. Focused tests prove anchor
selection and repaired guard accounting. The clean matching screen at
`runs/transformer-answer-v0.87-fullstack-baseline-floor-repaired-prompt-ownership-clean-smoke-dim4-context80/`
wrote the modern artifacts, recorded `562` active baseline prediction anchors,
`227` repair anchors, checked `50/50` steps, attempted `200` updates, ran `200`
one-step repairs, and rejected all `200` attempts as unsafe. Every recorded
snapshot preserved baseline/final QA and heldout target-token coverage at
`0.25`, predicted diversity at `3/8`, QA average target rank at `13.25`, and
heldout average rank at `13.375`. This is rejected evidence: post-update
baseline-covered repair is insufficient, and the next repair needs an objective
whose gradients preserve the floor before optimizer application.

v0.88 adds
`branch-balanced-context-profile-baseline-floor-objective-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
It puts balanced baseline-covered floor anchors into the same objective and
backward pass as the branch-diversity pressure. Focused tests prove anchor-batch
selection and objective guard accounting. The matching screen at
`runs/transformer-answer-v0.88-fullstack-baseline-floor-objective-prompt-ownership-smoke-dim4-context80/`
wrote the modern artifacts, recorded `562` active baseline prediction anchors,
`227` objective-side floor anchors, checked `50/50` steps, attempted `200`
updates, ran `200` objective anchor batches covering `2400` anchor records, and
rejected all `200` attempts as unsafe. Every recorded snapshot preserved
baseline/final QA and heldout target-token coverage at `0.25`, predicted
diversity at `3/8`, QA average target rank at `13.25`, and heldout average rank
at `13.375`. This is rejected evidence: coupling floor anchors with branch
pressure in one step is insufficient, which set up the stabilization-only
screen before branch-diversity pressure is added back.

v0.89 adds
`branch-context-profile-baseline-floor-stabilization-unlikelihood`.
It removes branch-diversity pressure from the attempted update shape and trains
only baseline-covered floor anchors under the same adaptive guard. Focused tests
prove the anchor-batch helper, stabilization guard accounting, objective catalog,
and replay-plan surfaces. The matching screen at
`runs/transformer-answer-v0.89-fullstack-baseline-floor-stabilization-smoke-dim4-context80/`
wrote the modern artifacts, recorded `562` active baseline prediction anchors,
`227` stabilization anchors, checked `50/50` steps, attempted `200` updates,
ran `200` stabilization anchor batches covering `2400` anchor records, and
rejected all `200` attempts as unsafe. Every recorded snapshot preserved the
branch-context gate and baseline/final QA and heldout target-token coverage at
`0.25`; deterministic verifier checks passed with no external model. This is
rejected evidence: even floor-only anchor updates are not yet accepted by the
guard, so the next repair should diagnose the guard/update interaction before
branch-diversity pressure is added back.

v0.90 adds baseline-floor rejection diagnostics to the same
`branch-context-profile-baseline-floor-stabilization-unlikelihood` screen. The
guard now records rejected update-shape counts, rejected learning-rate scale
counts, violation profile counts, a worst rejected coverage deficit, and compact
per-attempt floor diagnostics. The matching screen at
`runs/transformer-answer-v0.90-fullstack-baseline-floor-stabilization-diagnostics-smoke-dim4-context80/`
wrote the modern artifacts, checked `50/50` steps, attempted `200`
stabilization-only updates, rejected all `200`, and recorded rejected shape
counts `stabilization: 200` plus rejected scale counts of `50` each for `1`,
`0.25`, `0.05`, and `0.01`. Violation counts show `heldout` failed all `200`
attempts; `admissions`, `glossary`, and `qa` failed `150` each; `self` failed
`100`; and `learning` and `owner` failed `50` each. The worst measured floor
deficit is `0.25` on `learning` (`0.25 -> 0.0`). The verifier passed without an
external model, but promotion remains rejected on `branch_diversity_target`; the
next repair should use these profile-level diagnostics before adding branch
pressure back.

v0.91 adds
`branch-context-profile-baseline-floor-profile-targeted-stabilization-unlikelihood`.
It uses the v0.90 diagnostic result by replacing the random 32-anchor
stabilization batch with the full baseline-covered floor-anchor profile-target
surface. The matching screen at
`runs/transformer-answer-v0.91-fullstack-baseline-floor-profile-targeted-stabilization-smoke-dim4-context80/`
wrote the modern artifacts, recorded `227` floor anchors, requested a
profile-targeted batch size of `227`, covered `12` profile-target groups, and
ran `200` profile-targeted anchor batches covering `2400` anchor records. The
guard checked `50/50` steps, attempted `200` profile-targeted stabilization
updates, rejected all `200`, and accepted `0`. Rejected shape counts were
`profile_targeted_stabilization: 200`; each adaptive scale failed `50` times;
violation counts remained `heldout: 200`, `admissions: 150`, `glossary: 150`,
`qa: 150`, `self: 100`, `learning: 50`, and `owner: 50`; the worst deficit
remained `0.25` on `learning` (`0.25 -> 0.0`). The verifier passed without an
external model, but promotion remains rejected on `branch_diversity_target`;
full floor-anchor profile-target coverage alone is not the missing mechanic.

v0.92 adds
`branch-context-profile-baseline-floor-sequential-profile-stabilization-unlikelihood`.
It changes the repair shape from one full profile-target batch to sequential
source-profile floor repair with rollback after each unsafe profile group. The
matching screen at
`runs/transformer-answer-v0.92-fullstack-baseline-floor-sequential-profile-stabilization-smoke-dim4-context80/`
wrote the modern artifacts, recorded `227` floor anchors, requested a floor
batch size of `227`, covered `12` profile-target groups and `10` source-profile
groups, and ran `2000` sequential profile batches covering `2400` anchor
records. The guard checked `50/50` steps, attempted `200` sequential
stabilization updates, accepted `0`, rejected `200`, and recorded `200`
no-effective-update attempts because every source-profile group was rolled back
before the outer update could preserve the floor. Each adaptive scale failed
`50` times, and each source profile was rejected `200` times:
`bridge:owner`, `bridge:place`, `fact:learning`, `fact:owner`, `fact:place`,
`qa:glossary`, `qa:learning`, `qa:owner`, `qa:place`, and `qa:self`. The
verifier passed without an external model, but promotion remains rejected on
`branch_diversity_target`; sequential source-profile repair is not isolated
enough to create safe weight movement.

v0.93 adds
`branch-context-profile-baseline-floor-calibrated-sequential-profile-stabilization-unlikelihood`.
It extends v0.92 with calibrated adaptive scales
`1`, `0.25`, `0.05`, `0.01`, `0.0025`, `0.0005`, and `0.0001`, plus
coverage-only guard probes so floor-preservation checks do not compute
irrelevant representation geometry. The matching diagnostic screen at
`runs/transformer-answer-v0.93-baseline-floor-calibrated-sequential-profile-stabilization-step1-dim4-context80/`
wrote the modern artifacts, checked `1/1` direct step, attempted `5`
calibrated outer updates, rejected the four larger scales, and accepted the
first safe nonzero source-profile update at scale `0.0025`. The guard records
`50` sequential profile attempts, `1` accepted profile group
(`bridge:owner`), `49` rejected profile groups, `60` anchor records, `4`
no-effective-update attempts, and accepted update-shape counts
`calibrated_sequential_profile_stabilization: 1`. The verifier passed without
an external model, but promotion remains rejected on `branch_diversity_target`;
v0.93 proves calibrated sub-`0.01` movement can survive the baseline floor
guard, not that the transformer is ready for model-quality promotion.

v0.94 adds
`branch-context-profile-baseline-floor-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood`.
It extends v0.93 by searching the calibrated scale ladder separately for each
source-profile group, preserving the first safe profile-local update and
rolling back only unsafe profile-scale attempts. The matching diagnostic screen
at
`runs/transformer-answer-v0.94-baseline-floor-profile-scale-calibrated-sequential-stabilization-step1-dim4-context80/`
wrote the modern artifacts, checked `1/1` direct step, attempted `1` outer
profile-scale update, accepted it, and recorded `60` profile-scale attempts:
`8` accepted source-profile updates, `52` rejected profile-scale attempts, and
`72` anchor records. Accepted profile scales were `bridge:owner: 0.0025`,
`bridge:place: 0.0005`, `fact:learning: 0.0005`, `fact:owner: 0.0001`,
`fact:place: 0.0001`, `qa:glossary: 0.0001`, `qa:place: 0.0001`, and
`qa:self: 1`. The verifier passed without an external model, but promotion
remains rejected on `branch_diversity_target`; v0.94 proves profile-scale
memory expands safe floor-preserving movement beyond one source profile.

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

The latest unpromoted transformer diagnostic is
`runs/transformer-answer-v0.115.0-hidden-projection-margin-candidate-step1-dim4-context80/`.
v0.115.0 adds `branch-hidden-projection-margin-unlikelihood`, runs the one-step
candidate with output bias frozen, and records a directionally useful but
insufficient repair signal: average collapsed-token hidden advantage moves from
about `0.0842` to `0.0736`. Promotion remains
`blocked_before_quality_metrics` with `10/11` constraints passing and
`branch_diversity_target` failing. All `9/9` multi-target profiles are still
collapsed to `"n"`, `2` profiles still have zero target-token coverage, all
sampled profiles still have low representation separation, and hidden-projection
pressure remains primary across `9/9` profiles. It uses no external model, no
embeddings, no pretrained retriever, and no retrieval weight updates. v0.115.0
proves the repair surface is relevant, but a single branch batch is not enough
to earn promotion.

`runs/transformer-answer-v0.46-output-binding-rankscore-smoke-dim4-context80/`
tests that repair direction with `branch-output-binding-unlikelihood` and
rank-aware best-snapshot scoring. The run completed `20/20` direct steps with
output bias frozen. QA average target rank improved from `17.375` to `14.125`
and QA/heldout top-5 coverage reached `0.25`, but target-token coverage stayed
`0.0`, top-3 coverage ended `0.0`, and both profiles still collapsed to wrong
branch tokens. The repair is rejected for promotion, while rank-aware restore
remains a useful guardrail.

`runs/transformer-answer-v0.47-rank-margin-steps50-smoke-dim4-context80/`
adds `branch-rank-margin-unlikelihood`, which pushes each branch target above
the model's own top wrong tokens. The run completed `50/50` direct steps and
restored the rank-aware best snapshot from step `40`. QA average target rank
improved from `17.375` to `9.0`, target-token coverage rose to `0.125`, top-3
coverage rose to `0.25`, and top-5 coverage rose to `0.5`. This is the
strongest rank-lift evidence so far, but prediction diversity stayed `1/8` and
QA/heldout remained collapsed to wrong `"n"`, so it is not promotion evidence.

`runs/transformer-answer-v0.48-balanced-rank-margin-smoke-dim4-context80/`
combines target-balanced branch batches with rank-margin repair. It completed
`50/50` direct steps and reached QA predicted diversity `2/8`, average target
rank `9.375`, target-token coverage `0.125`, top-3 coverage `0.375`, and top-5
coverage `0.5`. It improves wrong-token diversity and top-3 coverage versus
v0.47, but QA and heldout still choose wrong top-1 branch tokens, so it remains
rejected evidence.

`runs/transformer-answer-v0.49-balanced-rank-margin-top1-smoke-dim4-context80/`
tests whether concentrating rank-margin pressure on only the top wrong token
converts rank lift into top-1 branch choices. It restored the best branch
snapshot from step `10`; QA target-token coverage stayed `0.125`, but average
target rank regressed to `12.5`, top-3 coverage fell to `0.125`, and top-5
coverage fell to `0.25`. The screen is rejected and points away from
single-wrong-token pressure.

`runs/transformer-answer-v0.50-balanced-topk-softmax-w5-smoke-dim4-context80/`
tests `branch-balanced-topk-softmax-unlikelihood`, a target-balanced branch
batch objective that makes the correct branch target compete in a restricted
softmax against the model's current top wrong tokens. It restored the best
branch snapshot from step `40`; QA target-token coverage stayed `0.125`, but
average target rank improved to `8.75`, top-3 coverage reached `0.375`, and
top-5 coverage reached `0.5`. The screen is stronger than v0.49 and comparable
to v0.48 on top-k coverage, but it is still rejected because QA and heldout
remain collapsed to one wrong top-1 branch token.

`runs/transformer-v0.51-foundation-stack-smoke/` is the latest transformer
foundation-stack evidence. v0.51 adds optimizer state and scheduling
mechanics, gradient accumulation, checkpoint-resume validation, v2 checkpoint
metadata, multi-head attention, RMSNorm, gated MLPs, tied output embeddings,
rotary-position support, cache-aware generation metadata, sampling controls,
token-level traces, and replayable eval sample JSONL before the next
direct-answer repair run. The all-switch smoke completed `2/2` language-model
steps with AdamW, wrote `optimizer_state.json`, saved a
`quarklm-transformer-v2` checkpoint, and produced traced eval artifacts. This
is mechanics-readiness evidence, not a promoted responder checkpoint.

`runs/transformer-answer-v0.52-fullstack-topk-softmax-smoke-dim4-context80/`
uses the full v0.51 stack for the first post-foundation direct-answer screen.
It reruns `branch-balanced-topk-softmax-unlikelihood` with AdamW, gradient
accumulation, two attention heads, RMSNorm, gated MLPs, tied output embeddings,
rotary positions, cache-aware metadata, and prompt-position projection. The
run completed `50/50` direct steps and restored the best branch snapshot from
step `0`. The baseline/final restored state had QA predicted diversity `3/8`,
target-token coverage `0.25`, average target rank `13.25`, top-3 coverage
`0.25`, and top-5 coverage `0.375`; heldout was similar with average rank
`13.375`. Training improved rank at later snapshots but collapsed target
coverage and diversity to one wrong token, so the screen rejects unchanged
top-k pressure under the full stack and points next toward bidirectional
prompt-to-token binding.

`runs/transformer-answer-v0.53-fullstack-bidir-binding-smoke-dim4-context80/`
adds `branch-balanced-bidirectional-binding-unlikelihood`, which binds prompt
contexts to target tokens in both directions. It completed `50/50` direct steps
under the same full stack and restored the best branch snapshot from step `40`.
QA average target rank improved to `7.875` with top-5 coverage `0.5`; heldout
average rank improved to `9.0` with top-5 coverage `0.375`. The screen is
still rejected for promotion because target-token coverage ended at `0.125`,
top-1 remained wrong, and the diversity target still failed `0/9` multi-target
profiles. This is partial rank-pressure progress and makes coverage
preservation the next repair target.

`runs/transformer-answer-v0.54-fullstack-coverage-binding-smoke-dim4-context80/`
adds `branch-balanced-coverage-binding-unlikelihood`, which combines
bidirectional binding with hard-wrong-token competition and an explicit
target-set mass coverage guard. The focused tests pass, but the full-stack
screen rejects the mechanism: it completed `50/50` direct steps and
best-snapshot scoring restored step `0`. Training snapshots improved QA rank as
far as `8.125`, but target-token coverage fell to `0.0` and top-1 collapsed to
wrong `"a"`. The guard prevented promotion of a worse checkpoint; the next
repair should preserve target-set coverage as its own objective before
sharpening exact target selection.

`runs/transformer-answer-v0.55-fullstack-target-set-coverage-smoke-dim4-context80/`
isolates target-set coverage with
`branch-balanced-target-set-coverage-unlikelihood`: no exact-target row loss,
no cross-context ownership term, and positive target CE disabled in the screen.
The focused tests pass, but the full-stack screen still restores step `0`.
Training snapshots improved QA average target rank to `10.0`, yet
target-token coverage again fell to `0.0` and top-1 collapsed to wrong `"a"`.
The failure is now sharper: batch-local target-set mass is not enough to
preserve eval target-token coverage, so the next repair should add explicit
anti-collapse pressure over predicted target tokens.

`runs/transformer-answer-v0.57-fullstack-target-diversity-smoke-dim4-context80/`
adds that pressure with `branch-balanced-target-diversity-unlikelihood`, which
combines target-set mass with a target-share diversity term. The focused tests
pass, including a regression that lifts both restricted target-set mass and the
weakest target's average share in a small branch batch. The full-stack screen
still rejects the mechanism: it completed `50/50` direct steps and
best-snapshot scoring restored step `0`. Training snapshots improved QA
average target rank to `10.0`, but target-token coverage again fell to `0.0`
with wrong `"a"` top-1 collapse. The next repair should preserve eval-wide
target-token coverage directly, likely through replay or diversity scoring tied
to heldout branch profiles rather than batch-local target sharing alone.

`runs/transformer-answer-v0.58-fullstack-target-replay-coverage-smoke-dim4-context80/`
implements that replay-shaped repair with
`branch-balanced-target-replay-coverage-unlikelihood`: the sampled branch batch
still trains unlikelihood, but target-set mass and target-share balance use the
broader admitted branch training pool at the same branch position. The focused
tests pass, including a regression where the sampled batch omits some pool
targets and training raises both replay target-set mass and the weakest missing
target share. The full-stack screen still rejects the mechanism: it completed
`50/50` direct steps and best-snapshot scoring restored step `0`. Training
snapshots improved QA average target rank as far as `6.875` and QA top-5
coverage to `0.5`, while admissions top-5 reached `0.5417`; however,
target-token coverage still hit `0.0` during training and QA/heldout top-1
predictions collapsed to one wrong branch token by step `50`. The next repair
should make replay context-owned, not merely pool-owned, so broad target mass
cannot be satisfied by the same target distribution on every branch context.

`runs/transformer-answer-v0.59-fullstack-context-replay-coverage-smoke-dim4-context80/`
implements context-owned replay with
`branch-balanced-context-replay-coverage-unlikelihood`: the repair batch still
trains wrong-token unlikelihood, while a broader target-balanced replay sample
trains each admitted branch context to own its own target within the replay
target set. The focused tests pass, including a regression that raises replay
target-set mass and the weakest owned-target share on fixed replay contexts.
The full-stack screen still rejects the mechanism: it completed `50/50` direct
steps and best-snapshot scoring restored step `0`. Training snapshots improved
QA average target rank as far as `7.375`, QA top-3 to `0.375`, QA top-5 to
`0.5`, and admissions top-5 to `0.5208` by step `50`; however, the diversity
target still failed `0/9`, target-token coverage hit `0.0` during training,
and no trained snapshot beat the step-0 baseline under branch snapshot scoring.
The next repair should either strengthen target-preserving ownership directly
or make snapshot scoring reject rank/top-k gains whenever target-token coverage
regresses.

`runs/transformer-answer-v0.60-fullstack-context-replay-coverage-floor-metadata-smoke-dim4-context80/`
implements the scoring side of that repair. Best branch snapshot selection now
has a profile-wise target-token coverage floor: every multi-target profile must
preserve its baseline coverage before rank/top-k improvements can promote a
trained snapshot. Direct-answer JSONL snapshots also record
`branch_target_coverage_by_profile`, making the floor auditable in run
artifacts. The focused tests pass, including a regression that rejects a
rank-lifted candidate when QA target-token coverage drops below baseline. The
full-stack screen completed `50/50` direct steps, wrote `7` clean direct-answer
JSONL rows, and restored step `0`. The baseline coverage floor was preserved
in the final row (`qa` `0.25`, `heldout` `0.25`, `admissions` `0.1429`,
minimum profile `0.0714`), while step `40` still improved QA average target
rank to `7.375` and top-5 to `0.5` only by regressing profile coverage. This
accepts the self-improvement gate repair but rejects the trained model
behavior; the next repair should make the objective preserve target-token
coverage under training.

`runs/transformer-answer-v0.61-fullstack-context-coverage-anchor-smoke-dim4-context80/`
moves coverage preservation into the objective with
`branch-balanced-context-coverage-anchor-unlikelihood`: replay branches whose
target is already the top-1 prediction receive an additional covered-target
anchor against the replay target set and hard wrong tokens. The focused tests
pass, including a regression where the anchor protects a covered branch better
than identical replay training without the anchor. The full-stack screen still
rejects the mechanism: it completed `50/50` direct steps, wrote `7`
direct-answer JSONL rows, and restored step `0` under the v0.60 coverage
floor. Training snapshots over-anchored the already-covered wrong `"i"` token:
QA/heldout predicted diversity fell to `1/8`, target-token coverage fell to
`0.125`, and average target rank regressed above `21`. The next repair should
make preservation target-balanced or profile-aware, so an anchor can protect
baseline coverage without turning one covered token into a global attractor.

`runs/transformer-answer-v0.62-fullstack-target-balanced-anchor-smoke-dim4-context80/`
makes the covered-target anchor target-balanced with
`branch-balanced-context-target-balanced-anchor-unlikelihood`: anchor losses are
averaged by covered target and skipped when a replay batch contains only one
covered target. The focused tests pass, including a regression where the
singleton guard skips the v0.61 over-anchor while the old global anchor still
raises that single token. The full-stack screen completed `50/50` direct
steps, wrote `7` direct-answer JSONL rows, and restored step `0` under the
v0.60 coverage floor. It avoided the v0.61 global `"i"` attractor, but
QA/heldout target-token coverage still collapsed to `0.0` during training and
trained snapshots remained ineligible for restore. The next repair should use
profile-level coverage deficits as the training signal, not only anchors from
already-covered replay branches.

`runs/transformer-answer-v0.64-fullstack-coverage-deficit-smoke-dim4-context80/`
adds that first deficit-driven training mode with
`branch-balanced-context-coverage-deficit-unlikelihood`: replay target tokens
that are absent from current replay predictions receive extra target-vs-hard
candidate pressure. The focused tests pass, including a regression that lifts a
missing replay target above the old context replay objective. The full-stack
screen completed `50/50` direct steps, wrote `7` direct-answer JSONL rows, and
restored step `0` under the v0.60 coverage floor. Step `50` reached QA branch
accuracy `1/8`, QA predicted diversity `4/8`, and QA average target rank
`10.0`, but QA and heldout target-token coverage regressed to `0.125`, so the
trained snapshots remained ineligible. The next repair should combine deficit
pressure with an explicit coverage-preserving constraint.

`runs/transformer-answer-v0.65-fullstack-coverage-preserving-deficit-smoke-dim4-context80/`
adds `branch-balanced-context-coverage-preserving-deficit-unlikelihood`, which
balances missing-target deficit pressure with target-balanced anchors for target
tokens that are currently represented in replay predictions. Focused tests
pass, including a regression where missing targets still lift and the
represented target is protected better than deficit-only training. The
full-stack screen completed `50/50` direct steps, wrote `7` direct-answer JSONL
rows, and restored step `0` under the v0.60 coverage floor. Step `50` improved
QA average target rank to `7.75`, heldout average target rank to `7.125`, and
top-5 coverage to `0.5`, but QA and heldout collapsed to predicted diversity
`1/8` around the represented `"i"` token and target-token coverage regressed to
`0.125`. The next repair should make preservation profile-aware instead of
anchoring current predicted target tokens.

`runs/transformer-answer-v0.67-profile-aware-replay-plan-smoke-dim4-context80/`
adds the first profile-aware replay-plan mechanics for that path. Branch replay
records can carry admitted source/profile keys, replay targets are partitioned
by profile for deficit and preservation accounting, and profile-aware modes
write `direct_answer_replay_plan.json` before direct-answer training. The
bounded smoke completed one gated branch-only direct step, wrote a plan for
`9144` branch records across `21` profiles, and showed separate profile floors
such as `qa:place` at `0.5` and `qa:color` at `0.0`. The branch-diversity
target still failed `0/9` multi-target profiles, so this is
mechanics-readiness evidence, not promotion evidence.

`runs/transformer-answer-v0.68-fullstack-profile-aware-preserving-deficit-smoke-dim4-context80/`
uses that profile-aware plan in the comparable full-stack repair screen. The
context gate passed, the replay plan covered `9144` branch records across `21`
profiles, and `50/50` direct steps completed with `7` direct-answer JSONL rows.
Step `40` improved QA average target rank to `6.5` and top-5 coverage to
`0.625`; heldout average rank improved to `6.875` with top-5 coverage `0.5`.
Those gains came with QA/heldout target-token coverage regressing to `0.125`
and predicted diversity collapsing to `1/8`, so best-snapshot scoring restored
step `0`. This rejects profile-aware preservation as sufficient by itself.

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

Branch diversity is now an explicit direct-answer snapshot target. Each branch
profile records target-token diversity, predicted-token diversity, target-token
coverage, dominant predicted token/rate, collapse status, and missing target
tokens. Each direct-answer snapshot also records a `branch_diversity_target`
summary across multi-target eval profiles. The context-80 smoke at
`runs/transformer-answer-v0.43-branch-diversity-target-smoke-dim4-context80/`
passed the branch-context gate, ran `5/5` direct steps, and then failed the
branch-diversity target across all `9` multi-target profiles. The final QA
profile had `target_unique: 8`, `predicted_unique: 1`, dominant token `"r"` at
rate `1.0`, and target-token coverage `0.125`. This turns branch diversity into
a required screen signal before a full greedy-eval promotion snapshot is worth
running.

The first diversity-aware training mode is now implemented as
`branch-diversity-unlikelihood`. It trains distinct branch targets together,
penalizes the model's current wrong prediction for each branch context, and
retains the existing branch-target contrast penalty. Unit coverage verifies that
the objective suppresses a global wrong token while raising target probability
on a small branch batch. The context-80 corpus smoke at
`runs/transformer-answer-v0.43-branch-diversity-train-smoke-dim4-context80/`
passed the branch-context gate and ran `10/10` direct steps, but the final
branch-diversity target still failed across all `9` multi-target profiles. QA
prediction moved from all `"x"` to all `"b"`, target-token coverage improved
`0.0 -> 0.125`, and `predicted_unique` stayed `1/8`. This is rejected
training-mode evidence: the objective moves the collapse token but does not yet
create prompt-specific branch diversity.

Direct-answer training can now exclude the transformer output bias from updates
with `--direct-answer-freeze-output-bias`. This tests whether branch-diversity
loss was escaping through one global output-bias move instead of learning
prompt-specific weights. Unit coverage verifies that branch-diversity training
can keep `bout` unchanged while still updating output weights. The context-80
corpus smoke at
`runs/transformer-answer-v0.43-branch-diversity-freezebias-smoke-dim4-context80/`
passed the branch-context gate and ran `50/50` direct steps with the output bias
frozen. Interval loss moved `3.6149 -> 3.5016`, but the final
branch-diversity target still failed across all `9` multi-target profiles. QA
prediction moved from all `"x"` to all `"w"`, final target-token coverage was
`0.0`, and `predicted_unique` stayed `1/8`. This is rejected stabilizer
evidence: output bias can be guarded, but the current direct-answer path still
collapses through deeper prompt-independent weights.

`branch-target-softmax-unlikelihood` now adds a restricted softmax over the
distinct branch targets in each batch, so each context's correct branch token
must beat the other observed target tokens directly. Unit coverage verifies
that the objective improves restricted target probability on a small branch
batch. The context-80 corpus smoke at
`runs/transformer-answer-v0.43-branch-target-softmax-freezebias-smoke-dim4-context80/`
passed the branch-context gate, froze output bias, and ran `50/50` direct
steps. Composite train loss moved `5.6671 -> 5.5820`, but the final
branch-diversity target still failed across all `9` multi-target profiles. QA
briefly reached `predicted_unique: 2` at step `20`, then collapsed back to all
`"w"` by step `50` with final target-token coverage `0.0`. This is rejected
target-set evidence: the restricted competition can crack collapse transiently,
but it does not yet stabilize prompt-specific branches.

Direct-answer runs can now opt into
`--direct-answer-restore-best-branch-snapshot`, which scores every branch
snapshot and restores the best measured branch-diversity checkpoint before
final metrics and checkpoint writing. The restore-best target-softmax smoke at
`runs/transformer-answer-v0.43-branch-target-softmax-restorebest-smoke-dim4-context80/`
passed the branch-context gate, froze output bias, ran `50/50` direct steps,
and restored the final checkpoint from step `40`. The final diversity target
still failed across all `9` multi-target profiles, but final QA target-token
coverage improved from the previous all-`"w"` final coverage `0.0` to all-`"u"`
coverage `0.125`. This is rejected guardrail evidence: best-snapshot
restoration preserves the best measured branch state, but it does not itself
create prompt-specific branch choices.

A prompt-focused representation screen added `--use-prompt-prefix-projection`,
a zero-initialized trainable projection over non-padding prompt-prefix positions
before the final answer token. The context-80 target-softmax restore-best smoke
at
`runs/transformer-answer-v0.43-prompt-prefix-target-softmax-restorebest-smoke-dim4-context80/`
moved all `20` prompt-prefix projection parameters and improved composite train
loss `5.6649 -> 5.5679`, but the final branch-diversity target still failed
across all `9` multi-target profiles. The final checkpoint restored from step
`40`; QA stayed collapsed to all `"u"` with target-token coverage `0.125` and
`predicted_unique` still `1/8`. This is rejected representation evidence:
targeted prompt-prefix access is active, but still not enough to separate
prompt-specific branches.

The next representation screen added `--use-prompt-position-projection`, which
keeps a separate zero-initialized trainable projection for each non-padding
prompt-prefix context position. The context-80 target-softmax restore-best
smoke at
`runs/transformer-answer-v0.43-prompt-position-target-softmax-restorebest-smoke-dim4-context80/`
moved `1108/1284` prompt-position projection parameters and improved composite
train loss `5.6649 -> 5.5679`, but the final branch-diversity target still
failed across all `9` multi-target profiles. The final checkpoint restored from
step `40`; QA stayed collapsed to all `"u"` with target-token coverage `0.125`
and `predicted_unique` still `1/8`. This is rejected representation evidence:
position-specific prompt access also moves, but still does not produce
prompt-specific branch choices.

A follow-up target-margin screen added `branch-target-margin-unlikelihood`, a
smooth pairwise margin loss over each batch's distinct branch targets. The
prompt-position context-80 smoke at
`runs/transformer-answer-v0.43-branch-target-margin-prompt-position-smoke-dim4-context80/`
passed the branch-context gate, froze output bias, ran `50/50` direct steps,
moved train loss `4.8973 -> 4.7784`, and moved `1108/1284` prompt-position
projection parameters. The final checkpoint restored from step `40`; QA stayed
collapsed to all `"u"` with target-token coverage `0.125`, `predicted_unique`
still `1/8`, and the branch-diversity target failed across all `9`
multi-target profiles. This is rejected target-margin evidence: pairwise target
separation improves the bounded loss but still does not stabilize
prompt-specific branch choices.

Direct-answer snapshots now include `branch_representation_profiles`, which
measure pairwise hidden-state distance between branch contexts before the
output head. A representation-contrast follow-up added
`branch-representation-contrast-unlikelihood`, which penalizes nearly identical
hidden states for different branch targets. The high-weight prompt-position
context-80 smoke at
`runs/transformer-answer-v0.43-branch-representation-contrast50-prompt-position-smoke-dim4-context80/`
used `--direct-answer-contrast-weight 50.0`, passed the branch-context gate,
froze output bias, ran `50/50` direct steps, and moved train loss
`53.5827 -> 53.4342`. The final checkpoint restored from step `40`; QA stayed
collapsed to all `"u"` with target-token coverage `0.125`, `predicted_unique`
still `1/8`, and average different-target hidden distance only about
`0.00107`. This is rejected representation-contrast evidence: the current
hidden states remain nearly indistinguishable at the answer branch.

A dim-8 capacity follow-up tested whether the representation-contrast path was
too narrow at embedding/feed-forward dimensions `4/8`. The matching 50-step
dim-8 screen reached step `40` but was too slow for the regular loop, so the
completed evidence run is
`runs/transformer-answer-v0.43-branch-representation-contrast50-prompt-position-smoke-dim8-context80-steps40/`.
It used embedding/feed-forward dimensions `8/16`, passed the branch-context
gate, froze output bias, ran `40/40` direct steps, and restored the final
checkpoint from step `10`. The restored QA different-target hidden distance
rose to about `0.00209`, but QA still collapsed to all `"u"` with target-token
coverage `0.125`, `predicted_unique` still `1/8`, and the branch-diversity
target failed across all `9` multi-target profiles. This is rejected capacity
evidence: more width gives more hidden distance, but not prompt-specific branch
choices.

A prompt-signal scale follow-up added `--prompt-position-projection-scale` so
bounded screens can amplify the prompt-position projection residual without
changing corpus data, tokenizer training, or initialization policy. The
scale-32 context-80 smoke at
`runs/transformer-answer-v0.43-prompt-position-scale32-repcontrast50-smoke-dim4-context80/`
passed the branch-context gate, froze output bias, ran `50/50` direct steps,
moved `1108/1284` prompt-position projection parameters, and restored the final
checkpoint from step `40`. The raw step-50 snapshot briefly pushed QA
different-target hidden distance to about `0.4115`, and the restored checkpoint
kept it above the prior dim-4 screen at about `0.01235`. QA still collapsed to
all `"u"` with target-token coverage `0.125`, `predicted_unique` still `1/8`,
and the branch-diversity target failed across all `9` multi-target profiles.
This rejects prompt-signal volume as the whole fix: the model can separate
hidden states more than before, but the output path still turns them into one
global branch token.

The next transformer checkpoint should include an open-source structure audit
before another repair objective is added. `STRUCTURE_AUDIT.md` records the
boundary: QuarkLM may study model/trainer/tokenizer/checkpoint patterns from
open-source language-model projects, but must not import pretrained weights,
pretrained tokenizer vocabularies, external embeddings, unledgered datasets, or
training text. The audit now includes a comparison table against common GPT
structure. It identifies the next implementation target as an opt-in
pre-layer-norm transformer block path with final normalization, preserving the
existing default path for checkpoint compatibility before the next
branch-diversity repair.

The opt-in pre-layer-norm path is now implemented with
`--use-pre-layer-norm`. It uses GPT-style pre-normalization for attention and
MLP sublayers and applies final layer normalization before the output head,
while leaving the legacy default path unchanged for older checkpoints and prior
evidence. Focused tests cover config round trip, older-checkpoint defaults,
scalar/float forward parity, parameter inclusion, and parser wiring. The
context-80 prompt-position representation-contrast smoke at
`runs/transformer-answer-v0.44-prelayernorm-repcontrast50-prompt-position-smoke-dim4-context80/`
passed the branch-context gate, froze output bias, ran `50/50` direct steps,
moved `1108/1284` prompt-position projection parameters and all `8` final-norm
parameters, and did not need best-snapshot restoration because step `50` was
the best measured branch snapshot. The final branch-diversity target still
failed across all `9` multi-target profiles; QA and heldout stayed fully
collapsed, with QA all `"y"` and target-token coverage `0.125`. The useful
change is that `7/9` profiles were no longer fully collapsed, including
admission paraphrases at `predicted_unique: 4/14` and admissions at
`2/14`. This is partial structural evidence, not promotion evidence: the next
repair should stabilize and extend that diversity to QA and heldout rather
than add another unrelated branch objective.

A target-balanced branch-batch follow-up tested whether repeated first-answer
tokens in the weighted direct-answer pool were crowding rare branch targets out
of representation-contrast updates. The new
`branch-balanced-representation-contrast-unlikelihood` mode builds each branch
batch from distinct target buckets while preserving the admitted-corpus-only
training boundary. The matching pre-layer-norm context-80 smoke at
`runs/transformer-answer-v0.44-target-balanced-prelayernorm-repcontrast50-prompt-position-smoke-dim4-context80/`
passed the branch-context gate and ran `50/50` direct steps, but every trained
snapshot scored worse than the baseline branch-diversity snapshot. Best-snapshot
restoration returned to step `0`; the restored final state collapsed all `9/9`
multi-target profiles to `"n"`, and QA stayed at `predicted_unique: 1/8` with
target-token coverage `0.125`. This rejects target-balanced branch sampling as
a standalone repair. The next repair should keep the pre-layer-norm path but
target the prompt-to-answer binding that keeps QA and heldout moving together.

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
