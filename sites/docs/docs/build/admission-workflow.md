---
title: Admission Workflow
description: Admit new knowledge before it can be trained.
---

# Admission Workflow

QuarkLM learns new facts through `corpus/admissions.jsonl`. A fact is not
training data just because someone typed it in chat. It becomes learnable only
after admission, curriculum regeneration, and a measured weight update.

Admit one fact:

```bash
PYTHONPATH=src python3 -m closed_world_lm.admit \
  --id learned-child-book \
  --person child \
  --object book \
  --color blue \
  --relation on \
  --container table
```

Admit a batch:

```bash
PYTHONPATH=src python3 -m closed_world_lm.admit \
  --batch path/to/new_admissions.jsonl
```

Duplicate ids are rejected before writing. When using default project paths,
direct and paraphrase admission probes are regenerated automatically.

## After Admission

Run a self-improvement cycle that compares against the last promoted report:

```bash
PYTHONPATH=src python3 -m closed_world_lm.self_improve answer-cycle \
  --run runs/self-improve-next \
  --compare-report runs/self-improve-v0.38/self_improvement_report.json
```
