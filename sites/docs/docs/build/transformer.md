---
title: Transformer
description: How the from-scratch QuarkLM transformer works, and its current status.
---

# Transformer

<p className="qlm-meta"><span>5 min read</span><span>For contributors</span><span>Updated 2026-06-19</span></p>

<div className="qlm-lead">

**What you will learn**

- What `transformer_char_model` is, and what it is built without.
- Why it is the weight-consolidation path, not the reliable answering path.
- The single gate that keeps it unpromoted: `branch_diversity_target`.
- How optional PyTorch parity attempts are recorded without promoting PyTorch.
- How to read the evidence — why `memory-served` and `weight-consolidated` are kept apart.

</div>

`transformer_char_model` is QuarkLM's from-scratch neural model: a tiny
decoder-only transformer introduced in v0.24, built without PyTorch, JAX, Hugging
Face, pretrained checkpoints, or pretrained tokenizers. It starts from random
weights and trains with a small standard-library scalar autodiff engine.

The scalar implementation is still the canonical reference. QuarkLM also has an
optional experimental PyTorch parity surface for tiny backend fixtures, but it
does not add PyTorch as a default dependency and it is not a promoted training
backend. It can only become performance evidence after deterministic scalar
parity gates pass for the relevant profile. Install the optional runtime extra
only when you want to attempt PyTorch parity:

```bash title="Install the optional PyTorch runtime"
python3 -m pip install -e ".[pytorch]"
```

Then use the runtime preflight before attempting that evidence:

```bash title="Check optional PyTorch runtime evidence"
PYTHONPATH=src python3 -m transformer_torch_runtime_report \
  --requested-device auto \
  --requested-dtype float64 \
  --output build/torch_runtime_report.json
```

The command exits nonzero when real PyTorch is unavailable, a test double is
detected, or the requested dtype cannot be used. A passing report only means the
runtime can attempt parity evidence; it does not promote PyTorch training.

```bash title="Record a PyTorch training parity attempt"
PYTHONPATH=src python3 -m transformer_torch_training_parity_attempt_cli \
  --requested-device cpu \
  --requested-dtype float64 \
  --output-dir build/torch_training_parity_attempt_float64
```

This command builds a scalar training fixture from the admitted curriculum,
constructs the optional PyTorch candidate, writes the training parity report,
and stores a compact attempt summary. The artifact set is:

| Artifact | Records |
| --- | --- |
| `scalar_training_fixture.json` | The scalar reference loss, logits, optimizer state, parameter manifest, and corpus-only tokenizer summary. |
| `torch_training_candidate.json` | The optional PyTorch candidate, runtime report, replay probes, and aggregate replay gate. |
| `training_parity_report.json` | The candidate-versus-scalar report, including runtime and replay-gate checks. |
| `torch_training_parity_attempt.json` | The concise decision summary, artifact paths, closed-world boundary flags, failed checks, backend-promotion gate, and next unsatisfied requirement. |

```bash title="Verify an existing PyTorch training parity attempt"
PYTHONPATH=src python3 -m transformer_torch_training_parity_attempt_cli \
  --output-dir build/torch_training_parity_attempt_float64 \
  --verify-existing
```

If real PyTorch is not installed, the command still writes the artifacts and
records `blocked_runtime_unavailable`. That blocked result is useful evidence;
it is not a model-quality failure and it does not change the scalar reference.
When a real PyTorch runtime is available, use `float64` for scalar parity
attempts; `float32` is useful later for performance experiments, but it can
introduce small numerical drift before the parity gate.

The optional real-runtime parity test is skip-safe under the default scalar
environment and passes only when PyTorch is installed and the replay evidence
matches scalar training:

```bash title="Run optional real PyTorch training parity test"
PYTHONPATH=src python3 -m unittest discover \
  -s tests \
  -p 'test_transformer_torch_real_runtime_parity.py'
```

The attempt summary also carries `training_backend_promotion_gate`. That gate is
expected to fail today: it records that matched replay parity is fixture-level
evidence, while a promoted or generalized PyTorch trainer still needs separate
model-quality, profile, and retention gates. When the closed-world boundary is
dirty, the gate names the exact failed boundary fields.
The attempt builder validates the summary before writing it, including attempt
status, next requirements, promotion gate, closed-world boundary flags, evidence
scope, artifact paths, and the fixture/candidate/report payload set. The stored
training parity report must match a report rebuilt from the paired fixture and
candidate, and the stored backend-promotion gate must match a gate rebuilt from
the candidate, report, and closed-world boundary. The next-requirements
diagnosis must also rebuild from the candidate runtime report, candidate, and
report. Standalone validation also checks the promotion-gate schema, check
catalog, blocker derivation, required future gate catalog, and compact corpus,
runtime, candidate, replay-gate, and report summary shapes. Written summaries
also carry SHA-256 payload hashes for sibling artifacts, and the persisted
corpus summary must match the scalar fixture and candidate backend corpus hash.
Written attempt directories are reloaded through the same validation contract
before the writer returns. The same command can run with
`--verify-existing` to audit a written attempt directory without
rebuilding it. Recorded artifact paths must resolve to the loaded files. The
optional public backend surface also exposes the written-attempt file map, hash
algorithm, hash builder, loader, and compact summary validator so contributors
can inspect the same persisted audit contract without reaching through private
module paths. Each `next_requirements` summary is a typed artifact with an
explicit kind and schema version, and that contract is available from the
optional public backend surface. The requirements artifact also has standalone
validation and a public stage catalog so next-action routing can be checked
without validating a full attempt bundle.
Stage/action consistency is validated too, so a well-shaped artifact cannot
route a replay blocker through a readiness or runtime action by mistake.
Runtime preflight actions come from a canonical status-to-action map that the
standalone validator also checks.
Runtime-preflight blockers use a paired status-to-blocker map so the
remediation action is tied to the failed runtime check that justifies it.

It is **not** the reliable answering path. Retrieval memory and the deterministic
responder already answer admitted probes exactly (see [Build](./index.md)). The
transformer is the *weight-consolidation* path — the component meant to gradually
learn language behavior from admitted candidates after memory has made the
knowledge available and evaluation can reject harmful updates.

:::note
The full version-by-version screen log (v0.24 to current) and every evidence
table live in [Transformer screen history](./transformer-screen-history.md).
This page is the durable explanation.
:::

## Architecture

The model is intentionally small, so cause and effect stay inspectable. Every
part is corpus-derived or randomly initialized.

<div className="qlm-grid">
<div><h4>Character tokenizer</h4><p><code>tokenizer.CharTokenizer</code> learns its vocabulary from admitted text and rejects out-of-vocabulary characters.</p></div>
<div><h4>Embeddings</h4><p>Learned token and position embeddings.</p></div>
<div><h4>Self-attention</h4><p>One causal self-attention block.</p></div>
<div><h4>Feed-forward</h4><p>One feed-forward block.</p></div>
<div><h4>LM head</h4><p>A next-character language-model head.</p></div>
<div><h4>Autodiff</h4><p>Dependency-free scalar autodiff.</p></div>
<div><h4>Initialization</h4><p>Random initialization only.</p></div>
<div><h4>Backend parity</h4><p>Optional PyTorch candidate artifacts checked against scalar fixtures.</p></div>
</div>

A pretrained vocabulary would cross the same boundary as pretrained weights —
see [Purity boundary](../secure/purity-boundary.md).

## Train a checkpoint

Run from the project root with `PYTHONPATH=src` set.

```bash title="Pretrain on the corpus"
# next-character language-model pretraining on the corpus
PYTHONPATH=src python3 -m transformer_char_model train \
  --run runs/transformer-smoke \
  --steps 40 --context-size 8 --embedding-dim 6 --feedforward-dim 12
```

```bash title="Evaluate a checkpoint"
# evaluate answer probes against a checkpoint
PYTHONPATH=src python3 -m transformer_char_model eval \
  --checkpoint runs/transformer-smoke/transformer.json \
  --json runs/transformer-smoke/transformer_eval.json
```

`answer-train` trains on corpus-derived answer lessons. It carries a large
catalog of direct-answer objectives aimed at the branch-diversity problem below;
each objective name and the screen that tested it is recorded in
[Transformer screen history](./transformer-screen-history.md). See
[Quickstart](./quickstart.md) for a representative `answer-train` invocation.

## The answer-training stack writes its own evidence

Every `answer-train` run emits machine-checkable artifacts, so a screen can be
audited rather than trusted:

| Artifact | Records |
| --- | --- |
| `experiment_intent.json` / `transformer_answer_metrics.json` | The screen's hypothesis, acceptance gate, and closing decision. |
| `training_plan.json` / `corpus_hygiene.json` | Source mixture, duplicate and train/eval overlap checks, candidate ratio, allowed sources. |
| `sweep_plan.json` | The controlled tokenizer, architecture, optimizer, and training-budget axes for the screen. |
| `replay_mixture_report.json` | New lessons, retained facts, glossary/self facts, unknown-policy probes, tokenizer stress strings, and heldout/paraphrase evidence. |
| `long_answer_diagnostics.json` | Longest eval answers with first drift, per-token NLL, token counts, generation timing, train timing, and candidate rank. |
| `candidate_quarantine.json` | Candidate lifecycle state; candidates are not training data until admitted to the ledger. |
| `closed_world_verifier.json` | Deterministic check that the data boundary, candidate exclusion, quarantine, and protected train/eval overlap pass. |
| `training_recipe.json` / `constraint_first_promotion.json` | Model, tokenizer, data, objective, optimizer, replay, and gates; blocks any loss, NLL, rank, or exact-quality number until constraints pass first. |
| `retrieval_memory_report.json` | Retrieval-memory evidence, kept separate from neural weight metrics. |

See [Transformer responsibilities](./transformer-responsibilities.md) for how
these surfaces are divided across modules.

## Current status: branch diversity is the blocker

The transformer is not promoted, and the docs say so plainly. Direct-answer
snapshots emit `branch_diversity_target`, which fails when multi-target eval
profiles collapse to too few predicted branch tokens — the model learns to
predict one dominant token instead of routing each prompt to its own answer.

From v0.112 onward the failure is classified as a critical `target_routing_gap`:
`9/9` multi-target profiles fail, representation separation across profiles is
low, and dominant-token wins are hidden-projection driven. Dozens of
direct-answer objectives (catalogued in the screen history) have moved coverage
and diagnostics forward without clearing the gate.

<div className="qlm-keypoint">

**Memory-served is not weight-consolidated**

Retrieval memory answers `219/219` eval probes exactly, with provenance and **no
weight updates**. That is evidence for the memory-first rail, not neural
promotion: `memory-served` is not `weight-consolidated`.

</div>

## Foundation-stack options

From v0.51, audited GPT-style components are available as opt-in flags before the
next repair objective: `--optimizer adamw`, gradient accumulation, warmup/decay
schedules, `--attention-heads`, `--use-rms-norm`, `--use-gated-mlp`,
`--tie-output-embeddings`, `--use-rotary-positions`, `--use-kv-cache-path`, and
`--use-pre-layer-norm`. Per `STRUCTURE_AUDIT.md`, QuarkLM may study open-source
model, trainer, tokenizer, and checkpoint *structure*, but must not import
external weights, tokenizers, embeddings, datasets, or training text.

The optimization track adds two explicit screens around that foundation:
`--transformer-profile modern_small`, which bundles pre-RMSNorm, RoPE,
multi-head attention, gated feed-forward, and AdamW-style optimizer settings as
an opt-in profile; and `--tokenizer closed-world-subword`, which writes
corpus-only tokenizer manifest and report artifacts. Neither screen promotes the
transformer by itself. They create controlled evidence for comparing baseline
character tokenization against guarded subword compression and modernized small
transformer mechanics. `answer-train` now accepts the same governed tokenizer
flags as classic transformer training, so a screen can vary tokenizer type
without leaving the closed-world manifest path. Each `answer-train` run writes
`sweep_plan.json` so that comparison axis is explicit before any result is
interpreted.

For cross-trial comparisons, use `answer-sweep`. It expands declared axes into
separate `answer-train` trial directories, preserves each trial's normal
evidence stack, and writes one `sweep_report.json` at the sweep root:

```bash title="Compare tokenizer trials"
PYTHONPATH=src python3 -m transformer_char_model answer-sweep \
  --run runs/transformer-answer-tokenizer-sweep \
  --steps 0 \
  --sweep-axis tokenizer=char,closed-world-subword
```

The sweep report is comparison evidence only. It cannot promote the transformer
while constraint-first promotion still rejects the underlying answer-training
trials.

## Current evidence

| Run | Signal | Value |
| --- | --- | --- |
| `runs/transformer-v0.25/` | Validation NLL | `3.5885 -> 3.4382` |
| `runs/transformer-v0.25/` | Answer exact eval | `0/28` |
| `runs/transformer-answer-v0.42/` | Direct-answer transformer exact | `0/219` |
| `runs/transformer-answer-v0.42/` | Selector / generator exact | `219/219` |
| `runs/transformer-answer-v0.42/` | Direct-answer target loss | `3.4278 -> 2.2708` |
| latest screens (through v0.115) | Promotion gate | rejected on `branch_diversity_target` |
| all runs | Pretrained weights / tokenizer / external embeddings | `false` |

The selector and generator reaching `219/219` while the direct transformer stays
at `0/219` is exactly why evidence states are kept separate: the system can
*serve* every answer while the neural weights have not yet *learned* to route
them. Full run-by-run detail is in
[Transformer screen history](./transformer-screen-history.md).

## What is next

<div className="qlm-next">

<a href="../transformer-screen-history/"><strong>Read next</strong><span>Transformer screen history</span><small>The version-by-version screen log and every evidence table.</small></a>

<a href="../transformer-responsibilities/"><strong>Read</strong><span>Transformer responsibilities</span><small>How the answer-training surfaces are divided across modules.</small></a>

<a href="../tokenizer/"><strong>Build</strong><span>Tokenizer</span><small>How the closed-world subword path is governed.</small></a>

<a href="../../secure/purity-boundary/"><strong>Concept</strong><span>Purity boundary</span><small>Why pretrained weights, tokenizers, and embeddings are out of bounds.</small></a>

</div>
