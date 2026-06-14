---
title: Experiment Registry
description: Run-intent artifacts for QuarkLM training evidence.
---

# Experiment Registry

v0.71 adds `src/closed_world_lm/experiment_registry.py` so training evidence
starts with an explicit intent instead of a loose command.

Every experiment intent records:

- version and run id;
- component under test;
- hypothesis;
- allowed data sources;
- planned artifacts;
- training recipe id;
- acceptance gates;
- failure criteria;
- notes;
- decision.

Self-improvement answer cycles write `experiment_intent.json` as soon as an
attempt directory exists, then close that intent with the promotion-gate result.
The final intent is copied into the attempt report and the latest run report.

Transformer answer-training runs also write `experiment_intent.json` before
training. They record baseline/final snapshot gates, closed-world data checks,
no-pretrained-weight/tokenizer/embedding checks, and direct-answer branch
screen gates when applicable. Until QuarkLM has a dedicated transformer
promotion gate, transformer runs close as rejected screen evidence rather than
promoted model evidence.

The registry is intentionally small: JSON artifacts, pure validation, and no
hidden promotion behavior.
