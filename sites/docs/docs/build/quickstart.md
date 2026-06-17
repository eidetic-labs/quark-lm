---
title: Quickstart
description: Run the current QuarkLM prototype, and read what each command produces.
---

# Quickstart

Every command runs from the project root with `PYTHONPATH=src` set, because the
Python modules live directly under `src/` and run as top-level modules. Nothing
here downloads weights or data — each step reads only the admitted corpus and
writes evidence under `runs/`.

## The mental model

A run moves left to right along the [learning path](./index.md): the corpus is
compiled into curriculum text, the deterministic rails answer from it, the
learned components train from random weights, and an audited cycle decides
whether anything is promoted.

```text
curriculum  ->  respond / retrieval  ->  answer_model / decoder / transformer  ->  self_improve
 (build text)    (no weight movement)         (gated weight updates)              (promote or reject)
```

## 1. Build the curriculum

```bash
PYTHONPATH=src python3 -m curriculum --output build
```

Regenerates `build/train.txt`, `build/valid.txt`, and manifest data from the
files named in `corpus/ledger.json`. Run this after any admission — everything
downstream reads these generated files, not the raw corpus.

## 2. Check the deterministic responder

```bash
PYTHONPATH=src python3 -m respond --eval --json runs/smoke/respond.json
```

`respond` is the grounded rail: it answers from the admitted corpus or returns
`unknown`. It moves no weights, so a green result here proves the corpus *can*
serve an answer — not that the transformer learned it.

## 3. Train the learned components (from random weights)

```bash
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

```bash
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

## 5. Run a full audited cycle

```bash
PYTHONPATH=src python3 -m self_improve answer-cycle \
  --run runs/self-improve-next \
  --compare-report runs/self-improve-v0.42/self_improvement_report.json
```

`self_improve` orchestrates training, evaluation, audits, and the run report,
comparing against the last promoted report so forgetting and coverage are
checked, not assumed.

## What "passing" means

The runs above are **smoke checks**: they prove the machinery runs. A run is only
*promoted* when it preserves the [purity boundary](../secure/purity-boundary.md),
passes the recorded promotion gate, and updates the docs that describe current
state — see [Release discipline](../operate/release-discipline.md). A run is not
promoted because it completed.
