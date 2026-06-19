# QuarkLM Forward Research Plan

Last updated: 2026-06-18.

## Purpose

v0.68 showed that QuarkLM can improve target-rank evidence while still erasing
profile coverage and branch diversity. That is not a small hyperparameter
problem. It is a process problem: the trainer can currently optimize a local
objective before the project has enough explicit machinery for experiment
intent, replay planning, verifier acceptance, corpus hygiene, and promotion
constraints.

This plan turns the next phase into a research-backed implementation sequence.
It combines published work, public open-source mechanics, and the current
QuarkLM codebase state. The goal is to keep building toward a closed-world
language model that trains only from QuarkLM's admitted dataset, improves its
weights through versioned runs, and eventually proposes improvements without an
external model shaping the training data.

## Research Method

The review used two kinds of sources:

1. Primary papers for the research claims.
2. Public project repositories and documentation for implementation mechanics.

The open-source projects are design references only. QuarkLM must not copy
their code, import their datasets, reuse pretrained weights, reuse pretrained
tokenizers, or use external model outputs as hidden training data.

## Research Synthesis

### Continual Learning

Continual learning surveys frame the central problem as plasticity without
forgetting. The 2024 LLM continual-learning survey categorizes updates across
continual pretraining, instruction tuning, and alignment, and highlights
benchmarks and evaluation as unresolved challenges
([Wu et al. 2024](https://arxiv.org/abs/2402.01364)). A newer 2026 survey
emphasizes rehearsal, regularization, architecture-based methods, forgetting
rates, and knowledge transfer metrics
([Chen et al. 2026](https://arxiv.org/abs/2603.12658)).

QuarkLM implication: replay and forgetting metrics are not optional side
reports. Every weight update should know which admitted profiles it is allowed
to improve, which prior profiles it must preserve, and which gates can reject
the update.

### Replay And Retention

Experience replay remains one of the most practical continual-learning
controls. Deep Generative Replay separates the solver from a generator of prior
experience ([Shin et al. 2017](https://arxiv.org/abs/1705.08690)).
Avalanche implements continual learning as streams, strategies, plugins,
buffers, and evaluation callbacks. Its docs show replay as a plugin that
changes the training dataloader from current examples alone to current examples
plus memory-buffer examples, with reservoir sampling and evaluation over the
test stream ([Avalanche training docs](https://avalanche.continualai.org/from-zero-to-hero-tutorial/04_training)).
Reverb makes the same point for RL systems: replay is an explicit data system
with insert and sample policies, not a private detail inside the loss
([Cassirer et al. 2021](https://arxiv.org/abs/2102.04736)).

QuarkLM implication: `direct_answer_replay_plan.json` is the right direction,
but replay planning should be extracted from `transformer_char_model.py` into a
standalone planner that can be tested, reused, and recorded before training.

### Self-Generated Data

Self-Instruct uses model generations to propose new instruction data, then
filters invalid or similar examples before fine-tuning
([Wang et al. 2022](https://arxiv.org/abs/2212.10560)). The public repo's
pipeline generates instructions, classifies task type, generates instances, and
then filters/processes/reformats them
([self-instruct repo](https://github.com/yizhongw/self-instruct)). Its README
also warns that a sampled quality analysis found many generated data points had
problems, which is exactly the risk QuarkLM must avoid.

STaR uses self-generated rationales only when the resulting answer is correct,
then fine-tunes and repeats ([Zelikman et al. 2022](https://arxiv.org/abs/2203.14465)).
Self-Refine uses a generator, feedback, and refiner loop at inference time
without weight updates ([Madaan et al. 2023](https://arxiv.org/abs/2303.17651);
[self-refine repo](https://github.com/madaan/self-refine)). Reflexion stores
trial feedback as linguistic memory rather than updating weights
([Shinn et al. 2023](https://arxiv.org/abs/2303.11366)).

QuarkLM implication: generated text must enter a candidate quarantine lane, not
the training corpus. "I learned something new" should mean a candidate was
proposed, verified against admitted sources, admitted with provenance, converted
to curriculum, trained into weights, and evaluated.

### Self-Judging Risk

Self-feedback can work, but it is fragile. A self-bias study found models tend
to favor their own outputs, and self-refinement can amplify that bias
([Xu et al. 2024](https://arxiv.org/abs/2402.11436)). Reward-hacking work shows
that a model evaluator and generator can drift apart from human judgment when
the evaluator is an imperfect proxy
([Pan et al. 2024](https://arxiv.org/abs/2407.04549)). The Self-Feedback
survey separates self-evaluation from self-update and asks when self-feedback
actually works ([Liang et al. 2024](https://arxiv.org/abs/2407.14507)).

QuarkLM implication: a future learned verifier is valuable, but it cannot be
the first gate. Start with deterministic closed-world verification and only
later train a verifier from admitted artifacts, accepted/rejected candidate
history, and run outcomes.

### Verifiable Rewards

Modern post-training increasingly uses verifiable rewards when the task has an
objective checker. Tulu 3 releases data, code, recipes, evaluations, and
decontamination alongside SFT, DPO, and reinforcement learning with verifiable
rewards ([Lambert et al. 2024](https://arxiv.org/abs/2411.15124);
[open-instruct repo](https://github.com/allenai/open-instruct)). DeepSeek-R1
shows that rule-based rewards can elicit reasoning behavior, but also that
multi-stage training and cold-start data were needed to address readability and
language-mixing failures
([DeepSeek-AI 2025](https://arxiv.org/abs/2501.12948)).

QuarkLM implication: our first verifier should be small and objective:
ledger membership, exact answer consistency, unknown-policy compliance, prompt
leakage absence, profile coverage, and branch diversity. We should not jump to
open-ended self-rewarding.

### Model Collapse

The model-collapse literature is directly relevant to a self-improving model.
Training recursively on model-generated data can remove tail information
([Shumailov et al. 2023](https://arxiv.org/abs/2305.17493)). A later study
argues collapse can be avoided when real and synthetic data accumulate instead
of synthetic data replacing real data
([Gerstgrasser et al. 2024](https://arxiv.org/abs/2404.01413)).

QuarkLM implication: original admitted records are permanent evidence. Generated
candidates can supplement after verification, but they must remain labeled by
origin, and replay must protect rare profiles.

### Small And Data-Centric Models

BabyLM findings show that small models can be trained more sample-efficiently,
but many curriculum-learning attempts were weak or only modestly useful
([Warstadt et al. 2025](https://arxiv.org/abs/2504.08165)). TinyStories shows
that constrained language distributions can teach small models coherent English,
but its dataset was generated by external models and therefore is a reference,
not a QuarkLM training source
([Eldan and Li 2023](https://arxiv.org/abs/2305.07759)). SmolLM2 is important
because it is explicitly data-centric: it uses staged mixtures, small-scale
ablations, manual mixture refinements, and releases prepared datasets
([Allal et al. 2025](https://arxiv.org/abs/2502.02737)).

QuarkLM implication: corpus quality and mixture planning matter more than
blindly increasing model complexity. Every training run should have a declared
mixture plan and an acceptance criterion.

### Transparent Open Models

LLM360 argues that open research needs more than final weights: code, data,
checkpoints, intermediate results, and analyses should be released together
([Liu et al. 2023](https://arxiv.org/abs/2312.06550)). OLMo releases training
data, training/evaluation code, model weights, checkpoints, and logs
([Groeneveld et al. 2024](https://arxiv.org/abs/2402.00838)). OLMo 2 goes
further with training recipes, logs, thousands of intermediate checkpoints, and
stage-specific data mixtures ([Team OLMo 2025](https://arxiv.org/abs/2501.00656)).
The OLMo README publishes stage-one and stage-two token counts, configs, seeds,
checkpoint links, and WandB links
([OLMo repo](https://github.com/allenai/OLMo)).

QuarkLM implication: rejected runs, configs, replay plans, and docs are not
overhead. They are the evidence trail required for the project thesis.

### Data Hygiene And Contamination

Deduplication can reduce memorized outputs and train-test overlap while
requiring fewer training steps for similar or better accuracy
([Lee et al. 2021](https://arxiv.org/abs/2107.06499)). Dolma exposes dataset
curation tooling, taggers, and fast deduplication as a first-class part of the
training stack ([Dolma repo](https://github.com/allenai/dolma)). Open-Instruct
ships decontamination scripts that index training prompts, query eval sets, and
produce contamination reports
([open-instruct decontamination](https://github.com/allenai/open-instruct/blob/main/decontamination/README.md)).
LatestEval reinforces that evaluation construction must actively avoid overlap
with training data ([Li et al. 2023](https://arxiv.org/abs/2312.12343)).

QuarkLM implication: corpus hygiene should become a mandatory artifact:
duplicates, near-duplicates, source mixtures, generated-candidate ratios,
protected-prompt overlap, and rare-profile coverage.

### Trainer And Recipe Boundaries

nanoGPT intentionally keeps a small model file and a small training loop; it
records config, optimizer state, best validation loss, and checkpoints in the
training script ([nanoGPT repo](https://github.com/karpathy/nanoGPT)). minGPT
separates a generic `Trainer` from model and dataset concerns
([minGPT repo](https://github.com/karpathy/minGPT)). GPT-NeoX exposes training,
evaluation, and generation entry points through YAML configs
([GPT-NeoX repo](https://github.com/EleutherAI/gpt-neox)). LLM Foundry uses
source modules, data-prep scripts, train/eval/inference scripts, YAML workflows,
and registries for components
([LLM Foundry repo](https://github.com/mosaicml/llm-foundry)). LitGPT presents
pretrain, continued pretrain, finetune, evaluate, deploy, and test as distinct
workflows ([LitGPT repo](https://github.com/Lightning-AI/litgpt)).

QuarkLM implication: `src/closed_world_lm/transformer_char_model.py` is now too
large for the next phase. It has model code, optimizer code, direct-answer
objectives, replay planning, snapshots, CLI parsing, checkpoint writing, and
run reporting in one file. The next implementation should extract recipe,
replay, evaluation, and artifact boundaries before adding more objective modes.

### Performance Backend Decision

Scalar Python remains QuarkLM's canonical reference implementation because it
keeps the model math inspectable and dependency-free. PyTorch is the planned
performance backend for scalable training, batched evaluation, optimized
attention, and hardware acceleration. PyTorch is allowed as a runtime library;
it does not change the closed-world boundary unless pretrained weights,
pretrained tokenizers, external embeddings, copied model code, or unledgered
data are introduced. NumPy is not a required interim backend and should only be
added later for a narrow diagnostic need.

QuarkLM implication: the scalar implementation should be optimized for
correctness, deterministic fixtures, and auditability rather than speed. The
PyTorch track should start as an experimental backend that must match scalar
logits, losses, and fixed-prompt generation on tiny parity fixtures before its
training runs can count as model-quality evidence.

The first implementation layer for that track is the dependency-free backend
policy and parity-fixture contract. Scalar fixtures record backend metadata,
model config, tokenizer summary, forward logits, losses, and fixed-prompt
generation traces. Candidate backends must compare against those fixtures before
their outputs can be trusted as model-quality evidence. This contract does not
add PyTorch as a dependency.

The second layer is an optional PyTorch backend surface: runtime availability
detection plus candidate parity artifacts. It records whether PyTorch is
installed, which device and dtype would be used, and why candidate cases are
blocked, pending, or matched.

The current experimental layer adds PyTorch-style forward parity through the
optional runtime surface. It covers the default scalar path plus post-layer
norm, pre-layer norm, pre-RMSNorm, gated MLP, multi-head attention,
rotary-position, deeper layer-stack, and tied output embedding fixtures.
Context-summary and prompt-projection variants are also covered by focused
fixtures with nonzero projection weights. KV-cache metadata equivalence is
covered by generation fixtures that compare cache events. Optimized cached
attention, real training, and optimizer behavior each require separate parity
gates before they can count as model-quality evidence.

The first training layer adds a scalar training parity fixture and report. It
captures initial weights, optimizer config, scalar step losses, final logits,
final loss, optimizer state, and a trained-parameter signature. This is still a
gate, not PyTorch training: a future PyTorch trainer must match the scalar
artifact before its weight updates can count as evidence.

The current training-backend layer adds a PyTorch training candidate artifact
that reports runtime availability, requested device and dtype, optimizer
config, and the scalar training case it would need to match. When PyTorch is
available but lacks required training capabilities, the candidate remains
`pending` with `training_runtime_incomplete`. When the runtime is
training-capable, the candidate still remains `pending` with
`training_not_implemented`; when the runtime or requested dtype is unavailable,
it records a blocked or pending case instead of fabricating metrics.

The current bridge records a trainable-parameter manifest for each scalar
training fixture and PyTorch candidate. The manifest names the scalar optimizer
parameter order, tensor shapes, contiguous optimizer-slot ranges, tied-output
status, and total trainable count. That keeps future PyTorch autograd work
accountable to the exact parameter mapping used by the scalar reference before
any optimizer state can be accepted as parity evidence.

The PyTorch training-readiness gate now checks runtime availability, requested
dtype support, parameter-manifest validity, autograd tensor construction, and
AdamW optimizer availability. Real PyTorch training, AdamW numerical parity,
accumulated-gradient parity, checkpoint compatibility, and final-loss parity
remain future work behind this gate.

The current trainable-state bridge builds PyTorch tensors from the scalar
fixture's initial weights by replaying the manifest names and shapes. Candidate
artifacts store only a JSON-safe state summary, not runtime tensors, so the
evidence trail can confirm tensor names, shapes, optimizer-slot ranges, and
`requires_grad` status without making PyTorch a required dependency. The
current initial-loss probe runs the tiny scalar fixture forward through those
tensors and records whether initial logits and loss match scalar evidence. The
current backward probe executes the tensor loss backward pass and reports
gradient coverage separately from optimizer behavior. The current optimizer-step
contract records the scalar schedule, per-parameter gradient clipping,
accumulation cadence, expected update steps, and final optimizer-state summary.
The current optimizer-step readiness probe validates that contract, maps
available `tensor.grad` values back to the trainable-parameter manifest, checks
gradient shapes and contiguous optimizer-slot coverage, and reports readiness
without applying an optimizer update. The current optimizer-step execution probe
then applies PyTorch value clipping to available `tensor.grad` values, records
before/after gradient extrema and changed-scalar counts, snapshots
trainable-parameter signatures around the optimizer call, instantiates PyTorch
AdamW when available, replays the scalar contract's accumulation cadence,
learning-rate schedule, and update/zero-grad calls, and records whether the
step-control trace matches the scalar step records. The probe also compares the
candidate post-step parameter signature against the scalar fixture's final
parameter signature and reports match or mismatch under the fixture tolerance.
It now also builds the scalar-expected AdamW post-update signature from the
current clipped gradients, assuming zero prior moments, and compares actual
post-step mutation against that expected update. A match here proves only local
current-gradient update math under those assumptions. The accompanying
gradient-accumulation report records the scalar pending/applied microstep
cadence, current gradient-sample signature, and reduction rule: scalar QuarkLM
applies AdamW to the mean of clipped microstep gradients. That means generic
PyTorch loss scaling is sufficient only when microstep clipping is inactive;
with clipping across accumulated microsteps, parity needs a clipped-gradient
buffer before the optimizer update. The report now includes PyTorch
accumulation-readiness requirements so replayed backward passes, loss scaling,
mean reduction, and clipped-gradient buffering are machine-checkable pending
items instead of implicit notes. Candidate artifacts also carry a PyTorch
accumulation replay plan: a per-microstep recipe for context, target, loss
scale, clipping, buffer action, reduction, optimizer step, and zero-grad
placement. The replay plan is not execution evidence; it explicitly marks
accumulated-gradient parity unproven until those backward passes run and match
scalar training evidence. The current replay-control probe runs the planned
microstep loss/backward control on a fresh tensor state and records that no
optimizer updates are applied. The replay-control probe now snapshots clipped
PyTorch microstep gradients and compares their signatures to scalar
clipped-gradient evidence. A mismatch is recorded as evidence, not promoted;
buffered-gradient, optimizer-update, final-logit, and final-loss parity remain
unproven. The replay-buffer comparison now folds replayed clipped gradients
through the scalar accumulation cadence and compares buffer-before,
buffer-after-add, and accumulated-gradient signatures to scalar evidence.
Buffer mismatches are recorded as blocking evidence, not promoted. That is
still not full PyTorch training parity: it does not yet prove optimizer
updates, final logits, final loss, or checkpoint compatibility. The next
implementation layer is matching those numerical update effects against scalar
training evidence. Scalar training fixtures now record per-step gradient-buffer
evidence: raw gradients, clipped gradients, buffer state before and after the
microstep, and the averaged gradient vector used when an update is applied. That
gives the PyTorch backend a concrete numerical target for accumulated-gradient
parity without promoting the PyTorch path yet.

## Current QuarkLM Diagnosis

### Strengths

- The project already preserves the closed-world boundary: no pretrained
  weights, no pretrained tokenizer, no external embeddings, and ledgered
  admissions.
- The self-improvement report already includes prompt leakage, forgetting,
  exact eval, promotion gate, corpus snapshots, corpus diffs, attempts archive,
  and deterministic diagnosis.
- v0.51 added a serious transformer mechanics stack: optimizer state, AdamW,
  scheduling, accumulation, multi-head attention, RMSNorm, gated MLPs, tied
  embeddings, rotary positions, prompt-position projection, checkpoint metadata,
  and eval traces.
- v0.67 added profile-aware replay-plan artifacts and tests.
- v0.68 proved the gate can reject rank improvements that damage coverage and
  diversity.

### Gaps

- `transformer_char_model.py` is a 9,494-line monolith. It is difficult to
  reason about new training modes because model, recipe, replay, eval,
  checkpoint, and CLI behavior are interleaved.
- v0.71 now records hypothesis, acceptance criteria, planned artifacts, and
  promotion decision before training begins.
- v0.72 now extracts replay planning from transformer objective plumbing into a
  reusable artifact module.
- v0.73 now writes corpus hygiene and training-plan artifacts for the current
  self-improvement and transformer run paths.
- v0.75 adds candidate quarantine artifacts and lifecycle states.
- v0.76 adds a deterministic closed-world verifier interface for candidate
  checks and training-plan approval.
- Snapshot scoring is stronger than before, but transformer promotion remains
  less integrated than the self-improvement promotion gate.
- The character tokenizer is still the purity baseline, but any future subword
  tokenizer needs an artifacted manifest, offsets, special-token policy, and
  admitted-corpus-only proof.

## Implementation Blueprint

### 1. Experiment Registry

Add an experiment-intent artifact before any new run:

- version;
- hypothesis;
- allowed data sources;
- planned artifacts;
- training recipe id;
- replay plan id;
- acceptance gates;
- failure criteria;
- result decision.

This turns each run from "try a mode" into an auditable experiment.

### 2. Corpus Governance Layer

Promote corpus state from implicit input to explicit training contract:

- admission ledger summary;
- original versus generated-candidate counts;
- duplicate and near-duplicate report;
- source/profile mixture;
- protected-prompt overlap;
- rare-profile coverage;
- tokenizer vocabulary proof.

### 3. Candidate Quarantine Layer

Generated lessons, generated probes, repair notes, and self-diagnosis proposals
should be written as candidates first. Candidate state should be one of:

- proposed;
- rejected;
- needs human review;
- verifier accepted;
- admitted;
- trained;
- promoted.

No candidate trains weights until it is admitted.

### 4. Closed-World Verifier Lane

Start deterministic:

- verify that candidate facts are entailed by admitted records;
- verify that answers match exact responder evidence;
- verify unknown-policy compliance;
- verify no protected eval prompt leaked into training text;
- verify profile coverage and replay preservation.

Later, train a small verifier from QuarkLM's admitted candidate history and run
outcomes. That learned verifier must remain advisory until it passes its own
held-out verifier evals.

### 5. Replay Planner

Extract replay from the transformer monolith into a reusable planner:

- input: examples, profiles, current snapshot, coverage floors, run objective;
- output: `replay_plan.json`;
- required fields: profile counts, target sets, represented targets, deficits,
  replay ratios, rare-profile floors, synthetic/original labels, and rejection
  constraints.

The trainer should consume the replay plan, not recreate it privately.

### 6. Training Recipe Layer

Add recipe objects that select:

- model family;
- tokenizer artifact;
- curriculum source;
- replay plan;
- objective;
- optimizer;
- snapshot cadence;
- promotion gates.

Recipes should be small and named. A recipe can fail, but it should be
reproducible.

### 7. Evaluation And Promotion Layer

Promotion should be constraint first:

- no prompt leakage;
- no unknown-policy regression;
- no forgetting regression;
- no protected held-out contamination;
- no profile coverage regression;
- no branch diversity regression;
- no rank-only promotion;
- no loss-only promotion.

Only after those constraints pass should rank, top-k, NLL, or loss be used to
compare snapshots.

### 8. Refactor Boundary

Keep the public CLI stable, but carve out narrow modules:

- `transformer_model.py` for model/config/checkpoint format;
- `transformer_training.py` for generic training loops;
- `transformer_replay.py` for replay records and plans;
- `transformer_eval.py` for branch profiles and promotion scoring;
- `experiment_registry.py` for run intent and decision artifacts;
- `candidate_quarantine.py` for candidate quarantine;
- `closed_world_verifier.py` for deterministic verifier checks.

The exact module names can change, but those responsibilities should not keep
living inside one file.

## Cross-Referenced Implementation Matrix

| External pattern | What others do | QuarkLM implementation |
| --- | --- | --- |
| OLMo and LLM360 transparency | Release data, recipes, logs, checkpoints, and intermediate results. | Treat every run artifact, rejected screen, replay plan, and docs update as part of the research claim. |
| Pythia learning dynamics | Same data order and frequent checkpoints make training dynamics analyzable. | Add experiment registry and deterministic data ordering for comparable screens. |
| Avalanche replay | Strategies and plugins modify dataloaders with memory buffers and evaluate after each experience. | Make replay a standalone planner with profile-aware memory, rare-profile floors, and retention gates. |
| Open-Instruct decontamination | Index train data and query eval data for overlap reports. | Add corpus hygiene reports for protected prompts and train/eval overlap. |
| Self-Instruct and STaR | Generate candidates, filter them, and train only accepted examples. | Add candidate quarantine plus deterministic verifier before admission. |
| SCoRe and self-bias work | Self-generated correction can collapse or amplify bias without good rewards. | Do not train on self-judged text until verifier quality is measured. |
| Tulu 3 and DeepSeek-R1 | Verifiable rewards work where correctness can be checked. | Use ledger, exact responder, unknown-policy, leakage, coverage, and diversity as first verifiable rewards. |
| SmolLM2 | Data mixtures are refined with ablations and stage evidence. | Add training-plan mixture reports and small ablation screens before bigger runs. |
| Hugging Face tokenizers | Tokenizers are trained artifacts with normalization, offsets, special tokens, padding, truncation, and serialization. | Keep character tokenizer now; require manifest and admitted-corpus-only proof before subword work. |
| nanoGPT, minGPT, GPT-NeoX, LLM Foundry, LitGPT | Keep model, trainer, config, recipes, eval, and data prep separable. | Refactor transformer recipe/replay/eval/artifact code out of the monolith before adding new modes. |

## Next Versions

### v0.69

Record this forward research plan and docs update. No model-quality claim.

### v0.70

Record the deeper cross-referenced research review in `DEEP_RESEARCH_REVIEW.md`
and the Docusaurus Learn section. No model-quality claim. This supersedes the
earlier v0.70 implementation target because the project needed a deeper
literature, implementation, and codebase gap review before writing more
mechanics.

### v0.71

Implemented experiment registry and run-intent schemas in
`src/closed_world_lm/experiment_registry.py`. Self-improvement answer cycles
and transformer answer-training runs now record hypothesis, allowed data,
planned artifacts, gates, failure criteria, and a result decision. Transformer
screens close through constraint-first promotion reports from v0.77 onward.

### v0.72

Implemented replay planning extraction in `src/closed_world_lm/replay_plan.py`
with focused tests. The existing profile-aware replay plan is behaviorally
preserved while replay records, profile grouping, coverage floors, and missing
target summaries now live outside the transformer monolith.

### v0.73

Implemented corpus hygiene and training-plan artifacts in
`src/closed_world_lm/corpus_hygiene.py`. Self-improvement and transformer
answer-training runs now write source mixture, duplicate, train/eval overlap,
candidate-ratio, rare-profile coverage, allowed-data, planned-artifact, and
replay-plan summary evidence.

### v0.74

Added `RESEARCH_IMPLEMENTATION_MAP.md` and the matching Docusaurus Learn page.
This research-control checkpoint cross-references papers, open-source
mechanics, and QuarkLM's current codebase into a source-to-gap-to-version
implementation map. No model-quality claim.

### v0.75

Implemented candidate quarantine artifacts in
`src/closed_world_lm/candidate_quarantine.py`. Self-improvement and transformer
answer-training runs now write `candidate_quarantine.json`, and training plans
record the quarantine path and summary. Candidates are not training data.

### v0.76

Implemented deterministic closed-world verifier checks in
`src/closed_world_lm/closed_world_verifier.py`. Self-improvement and
transformer answer-training runs now write `closed_world_verifier.json`, and
training plans embed verifier summaries. Verifier evidence is deterministic,
external-model-free, and focused on candidate checks plus training-plan
approval.

### v0.77

Implemented recipe objects and constraint-first promotion gates in
`src/closed_world_lm/training_recipe.py`. Self-improvement and transformer
answer-training runs now write `training_recipe.json` and
`constraint_first_promotion.json`, and transformer decisions cannot promote
from loss, rank, top-k, or NLL movement unless constraints pass first.

### v0.78

Implemented the first transformer responsibility split. Transformer
answer-training now uses `transformer_experiment.py` for artifact contracts,
intent/recipe surfaces, and promotion decisions; `transformer_training.py` for
JSONL snapshot writing, shuffled training cursors, and loss averaging; and
`transformer_objectives.py` for the direct-answer objective catalog.

### v0.79

Implemented transformer model/config and checkpoint metadata extraction.
`src/closed_world_lm/transformer_model.py` now owns model, optimizer, and
generation configs, validation, checkpoint identity, closed-world dataset
metadata, arg-to-config adapters, and run metadata.

### v0.80

Implemented transformer eval/checkpoint-load extraction.
`src/closed_world_lm/transformer_checkpoint.py` now owns checkpoint payload
loading and identity validation. `src/closed_world_lm/transformer_eval.py` now
owns probe loading, candidate collection, generic scoring, report assembly,
samples JSONL writing, and eval JSON writing.

### v0.81

Implemented the first post-surface anti-collapse objective:
`branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood`.
It adds balanced owned target-share pressure across replay targets inside each
profile-aware group, so the preserving-deficit objective no longer protects
only the currently represented target token.

### v0.82

Screened the v0.81 objective mechanic under the existing constraint-first gates
in `runs/transformer-answer-v0.82-fullstack-profile-target-share-smoke-dim4-context80/`.
The run writes the modern artifact set, fixes the transformer metrics
`external_embeddings` purity field, passes the verifier and branch-context
gate, and preserves target coverage after restore. It is rejected evidence:
trained snapshots still collapse QA and heldout branch diversity before any
rank gain can be trusted.

### v0.83

Implemented prompt-specific branch ownership with
`branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
The new loss adds a sibling-target margin inside each profile so a replay
context is trained to rank its own target above other profile targets. Focused
tests pass, and the full screen in
`runs/transformer-answer-v0.83-fullstack-prompt-ownership-smoke-dim4-context80/`
writes the modern artifact set. It is still rejected evidence: trained
snapshots improve QA rank to `8.625` only while collapsing QA and heldout to one
`"c"` branch token with `0.0` target-token coverage.

### v0.84

Implemented baseline replay anchors with
`branch-balanced-context-profile-baseline-anchored-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
The new mode preserves baseline replay predictions instead of following current
prediction drift during prompt-ownership training. Focused tests pass, and the
full screen in
`runs/transformer-answer-v0.84-fullstack-baseline-anchored-prompt-ownership-smoke-dim4-context80/`
records `562` active baseline prediction anchors. It is still rejected
evidence: trained snapshots improve QA rank to `8.0` and avoid the v0.83
`0.0` coverage collapse, but QA/heldout coverage only reaches `0.125`, below
the `0.25` baseline floor.

### v0.85

Implemented baseline-floor update gating with
`branch-balanced-context-profile-baseline-floor-gated-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
The new mode rejects an attempted direct-answer update whenever its branch
profile falls below the step-0 target-token coverage floor. Focused tests pass,
and the full screen in
`runs/transformer-answer-v0.85-fullstack-baseline-floor-gated-prompt-ownership-smoke-dim4-context80/`
records `562` active baseline prediction anchors and `50/50` rejected unsafe
updates. It is still rejected evidence: the guard preserves QA/heldout coverage
at `0.25`, but accepts no weight updates and branch diversity remains failed.

### v0.86

Implemented adaptive baseline-floor retries with
`branch-balanced-context-profile-baseline-floor-adaptive-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
The new mode restores model, optimizer, and RNG state before retrying the same
direct-answer update at learning-rate scales `1.0`, `0.25`, `0.05`, and `0.01`.
Focused tests pass, and the full screen in
`runs/transformer-answer-v0.86-fullstack-baseline-floor-adaptive-prompt-ownership-smoke-dim4-context80/`
records `562` active baseline prediction anchors and `200/200` rejected scaled
attempts. It is still rejected evidence: step-size retry alone does not produce
accepted safe updates.

### v0.87

Implemented baseline-floor repair retries with
`branch-balanced-context-profile-baseline-floor-repaired-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
The new mode keeps the adaptive retry guard, then applies one bounded
baseline-covered anchor repair before each failed retry is accepted or rejected.
Focused tests pass, and the clean full screen in
`runs/transformer-answer-v0.87-fullstack-baseline-floor-repaired-prompt-ownership-clean-smoke-dim4-context80/`
records `562` active baseline prediction anchors, `227` repair anchors, `200`
one-step repairs, and `200/200` rejected attempts. It is still rejected
evidence: post-update repair does not produce accepted safe updates.

### v0.88

Implemented objective-side baseline-floor anchors with
`branch-balanced-context-profile-baseline-floor-objective-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
The new mode adds a balanced floor-anchor batch to the same loss and backward
pass as branch-diversity pressure. Focused tests pass, and the full screen in
`runs/transformer-answer-v0.88-fullstack-baseline-floor-objective-prompt-ownership-smoke-dim4-context80/`
records `562` active baseline prediction anchors, `227` floor anchors, `200`
objective anchor batches, `2400` anchor records, and `200/200` rejected
attempts. It is still rejected evidence: the combined objective does not produce
accepted safe updates.

### v0.89

Implemented stabilization-only baseline-floor updates with
`branch-context-profile-baseline-floor-stabilization-unlikelihood`.
The new mode removes branch-diversity pressure from guarded attempts and trains
only baseline-covered floor anchors. Focused tests pass, and the full screen in
`runs/transformer-answer-v0.89-fullstack-baseline-floor-stabilization-smoke-dim4-context80/`
records `562` active baseline prediction anchors, `227` stabilization anchors,
`200` stabilization anchor batches, `2400` anchor records, and `200/200`
rejected attempts. It is still rejected evidence: floor-only stabilization
updates do not produce accepted safe updates under the current guard.

### v0.90

Implemented baseline-floor rejection diagnostics for the v0.89 stabilization
mode. The full screen in
`runs/transformer-answer-v0.90-fullstack-baseline-floor-stabilization-diagnostics-smoke-dim4-context80/`
records `200/200` rejected stabilization-shaped attempts, rejected scale counts
of `50` for each adaptive scale, violation profile counts, compact diagnostic
samples, and a worst rejected floor deficit of `0.25` on `learning`. It is still
rejected evidence for promotion, but it turns the next repair into a
profile-targeted floor repair instead of another blind objective change.

### v0.91

Implemented profile-targeted baseline-floor stabilization with
`branch-context-profile-baseline-floor-profile-targeted-stabilization-unlikelihood`.
The full screen in
`runs/transformer-answer-v0.91-fullstack-baseline-floor-profile-targeted-stabilization-smoke-dim4-context80/`
records `227` floor anchors, batch size `227`, `12` profile-target groups,
`200` profile-targeted anchor batches, `2400` anchor records, and `200/200`
rejected attempts. It is still rejected evidence: full baseline-covered
profile-target floor coverage does not change the v0.90 violation pattern.

### v0.92

Implemented sequential source-profile baseline-floor stabilization with
`branch-context-profile-baseline-floor-sequential-profile-stabilization-unlikelihood`.
The full screen in
`runs/transformer-answer-v0.92-fullstack-baseline-floor-sequential-profile-stabilization-smoke-dim4-context80/`
records `227` floor anchors, `12` profile-target groups, `10` source-profile
groups, `2000` sequential profile attempts, `2400` anchor records, and
`200/200` rejected outer attempts. It is still rejected evidence: every
source-profile group is rolled back, so the guard records `200`
no-effective-update attempts.

### v0.93

Implemented calibrated sequential source-profile baseline-floor stabilization
with
`branch-context-profile-baseline-floor-calibrated-sequential-profile-stabilization-unlikelihood`.
The diagnostic screen in
`runs/transformer-answer-v0.93-baseline-floor-calibrated-sequential-profile-stabilization-step1-dim4-context80/`
records calibrated scales down to `0.0001`, coverage-only guard probes, `50`
source-profile repair attempts, `49` profile-local rejections, and the first
accepted nonzero guarded update: `bridge:owner` at scale `0.0025`. It is still
rejected for model promotion because `branch_diversity_target` fails, but it
proves calibrated sub-`0.01` floor-preserving movement is possible.

### v0.94

Implemented profile-scale memory for calibrated sequential source-profile
baseline-floor stabilization with
`branch-context-profile-baseline-floor-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood`.
The diagnostic screen in
`runs/transformer-answer-v0.94-baseline-floor-profile-scale-calibrated-sequential-stabilization-step1-dim4-context80/`
records one accepted outer profile-scale update, `60` profile-scale attempts,
`8` accepted source-profile updates, `52` rejected profile-scale attempts, and
accepted profile scales spanning `1`, `0.0025`, `0.0005`, and `0.0001`.
Promotion remains rejected because `branch_diversity_target` still fails, but
safe calibrated movement now extends beyond one profile.

### v0.95+

The next transformer repair should translate expanded safe profile-scale
movement into branch-diverse improvement while preserving the same floor:
likely accepted-profile scoring, profile-specific scale memory across steps,
target-diversity-aware acceptance, or adapter-like repair surfaces before broad
branch-diversity pressure is added back.

## Stop Doing For Now

- Do not add another direct-answer mode as the first move.
- Do not run larger screens without an experiment intent artifact.
- Do not train on self-generated text without quarantine and verification.
- Do not treat retrieval, exact response, or candidate generation as proof of
  learned parametric behavior.
- Do not promote rank or loss improvements that reduce profile coverage,
  branch diversity, unknown-policy behavior, or retention.
- Do not replace admitted originals with summaries.
- Do not move to subword tokenization before the tokenizer artifact policy is
  written and tested.

## Decision

The next implementation phase should build the self-improvement operating
system before more objective tuning. QuarkLM needs experiment registry,
corpus-governance reports, candidate quarantine, verifier checks, replay
extraction, training recipes, and constraint-first promotion gates. That is the
structured path toward a model that can eventually say, truthfully and
audibly: "I learned something new, and now it is part of my training data."

## v0.112 Addendum

v0.112 adds `BRANCH_DIVERSITY_RESEARCH.md`, a matching Docusaurus Learn page,
and root-cause diagnostics under `branch_diversity_target.root_cause`. The
diagnostic screen in
`runs/transformer-answer-v0.112.0-branch-diversity-root-cause-profile-specific-memory-consolidation-step1-dim4-context80/`
keeps retrieval exact at `219/219`, rejects neural promotion on
`branch_diversity_target`, and classifies the final failure as a critical
`target_routing_gap`. The next implementation should audit logit priors,
output-bias escape paths, prompt-to-branch representation separation, and
profile/target imbalance before adding another branch objective.

## v0.113 Addendum

v0.113 adds that audit as `branch_routing_audit` in direct-answer snapshots.
The diagnostic screen in
`runs/transformer-answer-v0.113.0-branch-routing-audit-profile-specific-memory-consolidation-step1-dim4-context80/`
consumes the v0.112 plan, targets `owner`, `paraphrases`, and `learning`, keeps
retrieval exact at `219/219`, and remains rejected on
`branch_diversity_target`. The audit records
`routing_gap_requires_representation_and_logit_audit`, high output-bias escape
risk with `"n"` at bias rank `2`, low representation separation across `9/9`
multi-target profiles, and a `glossary` target-imbalance hotspot. The next
implementation should use those measurements to instrument dominant-token logit
priors and hidden-state separation before selecting another guarded repair
candidate.

## v0.114 Addendum

v0.114 adds `branch_logit_prior_profiles` and centroid separation metrics to
the direct-answer snapshots. The diagnostic screen in
`runs/transformer-answer-v0.114.0-logit-prior-representation-instrumentation-profile-specific-memory-consolidation-step1-dim4-context80/`
keeps retrieval exact at `219/219`, rejects promotion on
`branch_diversity_target`, and shows that dominant-token wins are
hidden-projection driven across `9/9` multi-target profiles even though
output-bias risk remains high. The next implementation should target guarded
hidden-projection or representation separation rather than another broad branch
objective.

## v0.115 Addendum

v0.115 adds `branch-hidden-projection-margin-unlikelihood` as the first
bias-frozen hidden-projection repair candidate. The candidate screen in
`runs/transformer-answer-v0.115.0-hidden-projection-margin-candidate-step1-dim4-context80/`
reduces average collapsed-token hidden advantage from about `0.0842` to
`0.0736`, but remains rejected on `branch_diversity_target`: all `9/9`
multi-target profiles still collapse to `"n"` and `2` profiles keep zero
target-token coverage. The next implementation should scale this repair beyond
one branch batch only under coverage-preserving promotion gates.
