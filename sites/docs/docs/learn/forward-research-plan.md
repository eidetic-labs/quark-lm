---
title: Forward Research Plan
description: The research-backed implementation sequence for QuarkLM's next self-improvement phase.
---

# Forward Research Plan

<p className="qlm-meta"><span>7 min read</span><span>For contributors</span><span>Last reviewed 2026-06-18</span></p>

<div className="qlm-lead">

**What you will learn**

- Why the v0.68 result moved the next step away from another direct-answer knob and toward an operating system around training.
- The seven operating-system mechanics, each one an auditable-evidence gate before a metric can move.
- Where the sequence stands: the operating system is implemented, the direct-answer objective is not promoted, and retrieval memory is a separate non-parametric rail.
- Why scalar Python remains the audit reference while PyTorch is the planned performance backend.
- The current decision and what the next repair must clear.

</div>

The full plan lives in the repository root at `FORWARD_RESEARCH_PLAN.md`.

This page records the strategy that set the direction for the current
operating-system work. v0.68 produced a useful but uncomfortable result: a
direct-answer change can improve target-rank evidence while damaging profile
coverage and branch diversity at the same time. The conclusion was that the next
step is not another direct-answer knob. The next step is the operating system
around training: experiment intent, corpus governance, candidate quarantine,
closed-world verification, replay planning, recipes, and constraint-first
promotion gates.

:::note
This page is the durable strategy. The version-by-version screen log it once
inlined now lives in
[Transformer screen history](../build/transformer-screen-history.md), which
records every objective name, run path, and evidence table. The same versions
are also summarized as a revised sequence in
[Deep research review](./deep-research-review.md).
:::

## What was reviewed

The v0.69 review cross-references three bodies of evidence:

- continual-learning and replay research;
- self-generated-data, self-feedback, and model-collapse research;
- public open-source mechanics from OLMo, Pythia, GPT-NeoX, nanoGPT, minGPT,
  LitGPT, LLM Foundry, Avalanche, Dolma, Open-Instruct, Self-Instruct,
  Self-Refine, and Hugging Face tokenizers.

Those sources are design references only. They do not change QuarkLM's purity
boundary: no pretrained weights, no pretrained tokenizer, no external
embeddings, no copied code, and no unledgered training data. See
[Purity boundary](../secure/purity-boundary.md).

v0.70 adds the deeper [Deep research review](./deep-research-review.md), which
cross-checks primary papers, official open-source mechanics, and the current
QuarkLM codebase before the next implementation step.

## Main finding

Mature language-model projects do not improve by quietly changing one training
knob at a time. They make data mixtures, recipes, replay buffers, evaluation
sets, contamination checks, checkpoints, logs, and release artifacts explicit,
so a result can be audited rather than trusted.

For QuarkLM, that finding has five consequences:

- generated lessons must be candidates before they are training data;
- replay must be planned before training, not reconstructed inside a loss;
- every run needs a hypothesis and an acceptance gate;
- deterministic verifier checks must precede any learned self-judgment;
- promotion must reject loss or rank gains that erase coverage, diversity,
  retention, or unknown-policy behavior.

<div className="qlm-keypoint">

**"Learned something new" is a reserved phrase**

This is the discipline that keeps "the model learned something new" from
becoming self-contamination. The phrase is reserved for material that has been
proposed, quarantined, verified, admitted to the ledger, trained, evaluated, and
promoted.

</div>

## Implementation sequence

The plan defines seven operating-system mechanics. Each one exists so a later
training screen cannot move a metric without first leaving auditable evidence.

<ol className="qlm-steps">
<li><strong><a href="../../operate/experiment-registry/">Experiment registry</a></strong><p>Record hypothesis, allowed data, planned artifacts, gates, failure criteria, and decision before every run.</p></li>
<li><strong>Replay extraction</strong><p>Move profile-aware replay planning out of the transformer monolith into a standalone planner, preserving prior behavior under focused tests.</p></li>
<li><strong><a href="../../operate/corpus-hygiene/">Corpus hygiene</a></strong><p>Report source mixtures, duplicate pressure, train/eval overlap, generated-candidate ratios, and rare-profile coverage.</p></li>
<li><strong><a href="../../operate/candidate-quarantine/">Candidate quarantine</a></strong><p>Store generated lessons, probes, and repair notes as candidates that cannot train weights until admitted to the ledger.</p></li>
<li><strong><a href="../../operate/closed-world-verifier/">Closed-world verifier</a></strong><p>Start deterministic; train a verifier later only from admitted candidate history and run outcomes.</p></li>
<li><strong><a href="../../operate/training-recipes/">Recipe layer</a></strong><p>Make model, tokenizer, curriculum, replay plan, objective, optimizer, snapshot cadence, and promotion gates named and reproducible.</p></li>
<li><strong>Constraint-first promotion</strong><p>Compare loss, rank, and top-k only after retention, leakage, unknown-policy, target coverage, and diversity pass.</p></li>
</ol>

## Performance backend decision

Scalar Python remains QuarkLM's canonical reference implementation because it
keeps the model math inspectable and dependency-free. PyTorch is the planned
performance backend for scalable training, batched evaluation, optimized
attention, and hardware acceleration. PyTorch is allowed as a runtime library;
it does not change the closed-world boundary unless pretrained weights,
pretrained tokenizers, external embeddings, copied model code, or unledgered
data are introduced. NumPy is not a required interim backend and should only be
added later for a narrow diagnostic need.

This is a roadmap decision, not a shipped capability claim. A PyTorch backend
must begin as experimental and earn trust against deterministic scalar parity
fixtures before its runs can count as model-quality evidence.

The first layer is the backend policy and scalar parity-fixture contract.
Scalar fixtures record backend metadata, model config, tokenizer summary,
forward logits, losses, and fixed-prompt generation traces. Candidate backends
must compare against those fixtures before their outputs can be trusted as
model-quality evidence. This contract does not add PyTorch as a dependency.

The second layer is an optional PyTorch backend surface: runtime availability
detection plus candidate parity artifacts. It records whether PyTorch is
installed, which device and dtype would be used, and why candidate cases are
blocked, pending, or matched.

The current experimental layer adds PyTorch-style forward parity through that
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
without promoting PyTorch training. The artifact records runtime availability,
requested device and dtype, optimizer config, and the scalar training case the
future trainer must match. If PyTorch is available but lacks required training
capabilities, the candidate is `pending` with `training_runtime_incomplete`.
If the runtime is training-capable, the candidate remains `pending` with
`training_replay_parity_pending` until every replay parity gate matches. If the
runtime or requested dtype is unavailable, it records a blocked or pending case
rather than fabricated metrics.

The current bridge records a trainable-parameter manifest on scalar training
fixtures and PyTorch candidates. It names the scalar optimizer parameter order,
tensor shapes, contiguous optimizer-slot ranges, tied-output status, and total
trainable count. That makes the future PyTorch trainer prove it is updating the
same parameter set as the scalar reference before optimizer state or weight
updates can be accepted.

The PyTorch training-readiness gate now checks runtime availability, requested
dtype support, parameter-manifest validity, autograd tensor construction, and
AdamW optimizer availability. Real PyTorch training, AdamW numerical parity,
accumulated-gradient parity, checkpoint compatibility, and final-loss parity
remain future gates.

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
through the scalar accumulation cadence only after replay microsteps exactly
align with scalar step records. It then compares buffer-before,
buffer-after-add, and accumulated-gradient signatures to scalar evidence.
Missing or misordered replay steps and buffer mismatches are recorded as
blocking evidence, not promoted. Scalar training fixtures now also record a
final trainable-parameter signature in manifest order. The replay-update
comparison is gated behind buffer parity: if the buffer comparison passes, it
applies the replayed accumulated gradient through AdamW on a fresh tensor state
and compares the post-update trainable signature to scalar evidence. A match
proves optimizer-update parity only; final logits and final loss remain
separate gates. The replay final-evaluation probe is gated behind
optimizer-update parity: after a
matched replay update, it computes final logits and final loss from a fresh
replay-updated tensor state and compares them to scalar evidence. A match proves
final-evaluation parity only. The replay checkpoint-compatibility probe is
gated behind final evaluation: it converts the replay-updated tensor state into
the existing QuarkLM checkpoint payload, validates and reloads it, then compares
the reloaded final logits and loss to scalar evidence. A match proves checkpoint
compatibility only; promoted PyTorch training remains a separate gate.
The aggregate training replay parity gate now checks runtime readiness, initial
loss, backward coverage, optimizer control, replay gradient signatures, replay
buffer parity, replay update parity, final evaluation, and checkpoint
compatibility together. The PyTorch candidate may move from pending to matched
only when all checks pass, and a matched candidate still records replay-parity
evidence rather than a promoted general training backend. The training parity
report now includes this aggregate gate as a required PyTorch check. The gate is
status-aware for replay buffer, update, final-evaluation, and checkpoint
probes, so `passed: true` cannot bypass the expected matched status or replay
evidence. It also requires replay-control count consistency: planned, executed,
backward, matched-gradient, mismatched-gradient, and microstep-record counts
must agree before replay gradients can count.
Runtime evidence also records whether the imported module is real PyTorch, a
test double, or unavailable. Test doubles can keep unit wiring deterministic,
but they cannot satisfy the aggregate replay parity gate or produce
model-quality training evidence. The `quark-lm-torch-runtime` preflight writes
that runtime evidence before any real PyTorch parity attempt, and PyTorch
candidate artifacts embed the same runtime report alongside their forward or
training evidence. Backend and training parity reports now verify that embedded
runtime report; training parity additionally requires runtime evidence that
allows a real PyTorch parity attempt. That runtime preflight does not prove
model-quality training evidence or promote the PyTorch backend.

## Where the sequence stands

The seven mechanics are built. The direct-answer objective they were meant to
unblock is not.

**Operating system (v0.71–v0.80): implemented.** v0.71 added the experiment
registry and run-intent schemas. v0.72 extracted replay planning into
`src/replay_plan.py` while preserving the profile-aware replay behavior. v0.73
added corpus-hygiene and training-plan artifacts. v0.74 added the
[Research implementation map](./research-implementation-map.md). v0.75 through
v0.80 added candidate quarantine, the deterministic closed-world verifier,
recipes and constraint-first promotion, and the transformer experiment,
artifact, model, config, checkpoint, and eval surfaces.

**Direct-answer screens (v0.81 onward): implemented, not promoted.** With the
operating system in place, the project resumed anti-collapse objective work
under the narrower surfaces. Every screen from v0.81 forward records an
experiment intent, corpus plan, replay plan, verifier checks, and explicit
promotion constraints, and every one is rejected on `branch_diversity_target`.
The objectives have moved coverage and diagnostics forward — baseline-floor
gating, adaptive and calibrated scale search, frontier and coverage-recovery
anchors, collapsed-profile binding, root-cause and routing audits, and the
v0.115 hidden-projection margin candidate — without clearing the gate. The
full objective-by-objective record is in
[Transformer screen history](../build/transformer-screen-history.md).

**Retrieval memory (v0.105.0): a separate, non-parametric rail.** v0.105.0 added
corpus-only retrieval memory. It builds `497` cards from the closed corpus and
answers `219/219` eval probes exactly, with provenance and **no weight updates,
no external model, and no external embeddings**. That is evidence that the
corpus contains the answers and that memory can serve them — not evidence that
the transformer has learned them. From v0.106.0 onward, retrieval success is
used to rank consolidation targets for gated training without counting a
retrieved answer as a learned weight.

<div className="qlm-keypoint">

**`memory-served` is not `weight-consolidated`**

Retrieval answering every admitted probe proves the corpus contains the answers
and that memory can serve them. It does not prove the transformer learned them.
The two are tracked as distinct evidence states.

</div>

## Current decision

v0.69 is strategy evidence and v0.70 is deep-research evidence. The
operating-system steps (v0.71–v0.80) and the screens that follow are
implementation and diagnostic evidence. None of them are model-quality promotion
evidence.

The from-scratch transformer remains unpromoted, blocked on
`branch_diversity_target`: multi-target eval profiles still collapse to too few
predicted branch tokens. Retrieval memory answering every admitted probe does
not change that status. The next repair must improve branch diversity under the
existing constraint-first gates, without treating retrieved answers as learned
transformer weights and without relaxing the target-token, coverage, or
diversity gates.

<div className="qlm-next">
<a href="../../build/transformer-screen-history/"><strong>Read next</strong><span>Transformer screen history</span><small>Every objective name, run path, and evidence table behind the unpromoted gate.</small></a>
<a href="../deep-research-review/"><strong>Read next</strong><span>Deep research review</span><small>The v0.70 cross-check of primary papers, open-source mechanics, and the codebase.</small></a>
<a href="../research-implementation-map/"><strong>Read next</strong><span>Research implementation map</span><small>Where each reviewed source landed in the implemented operating system.</small></a>
</div>
