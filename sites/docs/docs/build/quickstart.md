---
title: Quickstart
description: Run the current QuarkLM prototype, and read what each command produces.
---

# Quickstart

<p className="qlm-meta"><span>10 min read</span><span>For first-time operators</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will do**

Build the curriculum, check the deterministic responder, train the learned
components from random weights, run an answer-training screen, and run a full
audited cycle. Nothing here downloads weights or data — each step reads only
the admitted corpus and writes evidence under `runs/`, so you can inspect it
after the run.

</div>

Every command runs from the project root with `PYTHONPATH=src` set, because the
Python modules live directly under `src/` and run as top-level modules.

## The mental model

A run moves left to right along the [learning path](./index.md): the corpus is
compiled into curriculum text, the deterministic rails answer from it, the
learned components train from random weights, and an audited cycle decides
whether anything is promoted.

```text title="The run, left to right"
curriculum  ->  respond / retrieval  ->  answer_model / decoder / transformer  ->  self_improve
 (build text)    (no weight movement)         (gated weight updates)              (promote or reject)
```

<div className="qlm-keypoint">

**Memory-served is not weight-consolidated**

The deterministic rails answer *from* the corpus; the learned components answer
*after* training on it. A green result on the rails proves the corpus can serve
an answer — it does not prove any model learned it.

</div>

## Before you start

- You are at the project root, where `corpus/ledger.json` lives.
- `PYTHONPATH=src` is set, so the modules under `src/` resolve as top-level
  modules.
- You can write under `runs/` — every step records its evidence there.

## 1. Build the curriculum

```bash title="Compile the corpus into curriculum text"
PYTHONPATH=src python3 -m curriculum --output build
```

Regenerates `build/train.txt`, `build/valid.txt`, and manifest data from the
files named in `corpus/ledger.json`. Run this after any admission — everything
downstream reads these generated files, not the raw corpus.

## 2. Check the deterministic responder

```bash title="Run the grounded rail and write its result"
PYTHONPATH=src python3 -m respond --eval --json runs/smoke/respond.json
```

`respond` is the grounded rail: it answers from the admitted corpus or returns
`unknown`. It moves no weights, so a green result here proves the corpus *can*
serve an answer — not that the transformer learned it.

## 3. Train the learned components (from random weights)

```bash title="Short smoke runs for each learned component"
PYTHONPATH=src python3 -m answer_model train --run runs/answer-smoke
PYTHONPATH=src python3 -m answer_decoder train --run runs/decoder-smoke
PYTHONPATH=src python3 -m transformer_char_model train \
  --run runs/transformer-smoke --steps 20 --context-size 8
```

Each starts from random initialization and writes a versioned checkpoint under
its `--run` directory. These are short **smoke** runs — enough to confirm the
training loop works, not to promote anything.

## 4. Run an answer-training screen

`answer-train` is where the transformer is pushed to *route* prompts to their
answers. It carries a large catalog of direct-answer objectives (see
[Transformer](./transformer.md)); a representative invocation:

```bash title="A representative answer-training screen"
PYTHONPATH=src python3 -m transformer_char_model answer-train \
  --run runs/transformer-answer-smoke \
  --steps 100 --eval-every 0 \
  --candidate-scope eval \
  --selector-steps 200 --selector-eval-every 0 --selector-emit-completions \
  --generator-steps 400 --generator-eval-every 0 \
  --direct-answer-steps 100 --direct-answer-eval-every 0 \
  --direct-answer-mode periodic-balanced-repair-unlikelihood \
  --direct-answer-negative-weight 1.0 --direct-answer-positive-weight 1.0 \
  --direct-answer-rollout-interval 50
```

This writes the full evidence stack — `training_plan.json`,
`closed_world_verifier.json`, `constraint_first_promotion.json`, and the rest
(see [Transformer](./transformer.md)). The objective flags are catalogued in
[Transformer screen history](./transformer-screen-history.md).

<div className="qlm-keypoint">

**The transformer is not promoted**

The answer-training screen exercises the learned path, but a passing screen does
not promote the transformer. It stays blocked on `branch_diversity_target`
regardless of how the smoke run completes.

</div>

## 5. Run a full audited cycle

```bash title="Run a full cycle and compare against the last promoted report"
PYTHONPATH=src python3 -m self_improve answer-cycle \
  --run runs/self-improve-next \
  --compare-report runs/self-improve-v0.42/self_improvement_report.json
```

`self_improve` orchestrates training, evaluation, audits, and the run report,
comparing against the last promoted report so forgetting and coverage are
checked, not assumed.

## What you produced

If you walked the full path above, `runs/` now holds the evidence for each
stage:

<div className="qlm-grid">

<div>
<h4><code>runs/smoke/respond.json</code></h4>
<p>The deterministic responder's evaluation result — the grounded rail answering from the corpus or returning <code>unknown</code>.</p>
</div>

<div>
<h4><code>runs/*-smoke/</code></h4>
<p>Versioned checkpoints for <code>answer_model</code>, <code>answer_decoder</code>, and the transformer, each started from random weights.</p>
</div>

<div>
<h4><code>runs/transformer-answer-smoke/</code></h4>
<p>The full evidence stack — <code>training_plan.json</code>, <code>closed_world_verifier.json</code>, <code>constraint_first_promotion.json</code>, and the rest.</p>
</div>

<div>
<h4><code>runs/self-improve-next/</code></h4>
<p>The audited-cycle run report, compared against the last promoted report so forgetting and coverage are checked.</p>
</div>

</div>

## What "passing" means

The runs above are **smoke checks**: they prove the machinery runs. A run is only
*promoted* when it preserves the [purity boundary](../secure/purity-boundary.md),
passes the recorded promotion gate, and updates the docs that describe current
state — see [Release discipline](../operate/release-discipline.md). A run is not
promoted because it completed.

:::note
Docs are a promotion gate, not training input. Updating the page that describes
current state is part of what makes a run promotable — it is never folded back
into the corpus.
:::

## What is next

<div className="qlm-next">

<a href="./transformer.md">
<strong>Read next</strong>
<span>The transformer</span>
<small>How the from-scratch model works and why it stays blocked on branch_diversity_target.</small>
</a>

<a href="../secure/purity-boundary.md">
<strong>Read next</strong>
<span>The purity boundary</span>
<small>The separation a run must preserve before anything is promoted.</small>
</a>

<a href="../operate/release-discipline.md">
<strong>Step up</strong>
<span>Release discipline</span>
<small>What turns a smoke check into a promoted run.</small>
</a>

</div>
