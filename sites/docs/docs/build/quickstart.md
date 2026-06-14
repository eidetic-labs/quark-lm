---
title: Quickstart
description: Run the current QuarkLM prototype.
---

# Quickstart

Run commands from the project root:

```bash
PYTHONPATH=src python3 -m closed_world_lm.curriculum --output build
PYTHONPATH=src python3 -m closed_world_lm.respond --eval --json runs/smoke/respond.json
PYTHONPATH=src python3 -m closed_world_lm.answer_model train --run runs/answer-smoke
PYTHONPATH=src python3 -m closed_world_lm.answer_decoder train --run runs/decoder-smoke
PYTHONPATH=src python3 -m closed_world_lm.transformer_char_model train \
  --run runs/transformer-smoke \
  --steps 20 \
  --context-size 8
PYTHONPATH=src python3 -m closed_world_lm.transformer_char_model answer-train \
  --run runs/transformer-answer-smoke \
  --steps 100 \
  --eval-every 0 \
  --candidate-scope eval \
  --selector-steps 200 \
  --selector-eval-every 0 \
  --selector-emit-completions \
  --generator-steps 400 \
  --generator-eval-every 0 \
  --direct-answer-steps 100 \
  --direct-answer-eval-every 0 \
  --direct-answer-mode periodic-balanced-repair-unlikelihood \
  --direct-answer-negative-weight 1.0 \
  --direct-answer-positive-weight 1.0 \
  --direct-answer-rollout-interval 50
```

For a full audited cycle:

```bash
PYTHONPATH=src python3 -m closed_world_lm.self_improve answer-cycle \
  --run runs/self-improve-next \
  --compare-report runs/self-improve-v0.38/self_improvement_report.json
```

The short runs are smoke checks. Promoted runs should use the release discipline
in [Operate](../operate/release-discipline.md).
