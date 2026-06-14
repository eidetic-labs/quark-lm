---
title: Generated Probes
description: Probes generated from admitted memory and glossary sources.
---

# Generated Probes

Admission probes are generated from `corpus/admissions.jsonl` so admitted-memory
evals do not drift from admitted facts. Glossary probes are generated from the
`probe_words` list in `corpus/glossary.json` so definition evals do not drift
from the admitted glossary.

Check sync:

```bash
PYTHONPATH=src python3 -m closed_world_lm.admission_probes --check
PYTHONPATH=src python3 -m closed_world_lm.glossary_probes --check
```

Current generated files:

| File | Purpose |
| --- | --- |
| `evals/admissions.jsonl` | Direct place, color, owner, and training-data probes. |
| `evals/admission_paraphrases.jsonl` | Alternate surface forms for admitted facts. |
| `evals/glossary.jsonl` | Definition probes for configured glossary probe words. |

Current v0.38 counts are `48` direct probes and `84` paraphrase probes, generated
from `12` admitted facts, plus `38` glossary probes generated from `19` glossary
probe words.

The self-improvement report includes a combined `admission_probe_audit` with
direct and paraphrase results plus a separate `glossary_probe_audit`. Both must
pass before promotion.
