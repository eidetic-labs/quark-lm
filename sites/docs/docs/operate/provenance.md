---
title: Provenance
description: Corpus snapshots and diffs.
---

# Provenance

QuarkLM records corpus provenance in each self-improvement report.

`corpus_snapshot.json` captures:

- ledger source ids
- file paths
- training permissions
- curriculum-generation permissions
- file hashes
- JSONL record counts
- admitted memory ids

`corpus_diff.json` compares the current snapshot to the previous promoted run.
This makes corpus changes visible next to weight changes and eval changes.

v0.18 added `admission-paraphrase-probes-v0` as an eval-only source.

v0.19 changed the glossary, admitted-memory log, and generated admission probes
when `learned-ivy-stone` entered the corpus.

v0.20 added `glossary-probes-v0` as an eval-only source and changed the
glossary source to include explicit glossary probe words.

v0.21 added admitted memories `learned-noah-shell`, `learned-ava-coin`, and
`learned-omar-drum`; changed admission, admission-paraphrase, glossary, and
glossary-probe sources; and kept grammar unchanged.

v0.22 changed grammar, self-probe, and learning-probe sources to teach
self-diagnosis source, external-model-shaping policy, and report-evidence repair
selection. Admissions and generated admission/glossary probes stayed unchanged.

v0.23 added attempt archives under `attempts/attempt-###/`, preserving a failed
undertrained attempt and a repaired passing attempt in the same run directory.

v0.24 kept corpus sources unchanged from v0.23, passed the archived
self-improvement run, and added separate transformer architecture evidence under
`runs/transformer-v0.24/`.

v0.25 added admitted memories `learned-mia-ring`, `learned-leo-kite`, and
`learned-nina-bell`; added glossary probe words `ring`, `kite`, and `bell`;
changed admission, admission-paraphrase, glossary, and glossary-probe sources;
and added transformer architecture evidence under `runs/transformer-v0.25/`.

v0.26 kept corpus sources unchanged from v0.25, passed the archived
self-improvement run, and added transformer answer-lesson evidence under
`runs/transformer-answer-v0.26/`.

v0.27 kept corpus sources unchanged from v0.26, passed the archived
self-improvement run, and added context-48 transformer answer evidence plus a
faster eval-scoped candidate evaluator under `runs/transformer-answer-v0.27/`.

v0.28 added admitted memories `learned-sara-marble`, `learned-milo-spoon`, and
`learned-ruth-ribbon`; added glossary probe words `marble`, `spoon`, and
`ribbon`; changed admission, admission-paraphrase, glossary, and glossary-probe
sources; preserved two failed self-improvement attempts before passing
`attempt-003`; and added transformer prefix-choice evidence under
`runs/transformer-answer-v0.28-choice-prefix-pilot/`.

v0.29 kept corpus sources unchanged from v0.28, passed the archived
self-improvement run on `attempt-001`, and added transformer answer-selector
evidence under `runs/transformer-answer-v0.29-selector-fast/`.

v0.30 kept corpus sources unchanged from v0.29, passed the archived
self-improvement run on `attempt-001`, and added transformer selector-emission
evidence under `runs/transformer-answer-v0.30-selector-emission/`.

v0.31 kept corpus sources unchanged from v0.30, passed the archived
self-improvement run on `attempt-001`, and added no-candidate transformer-guided
answer-generator evidence under
`runs/transformer-answer-v0.31-generator-weighted-lr035-80k/`.

v0.32 kept corpus sources unchanged from v0.31, passed the archived
self-improvement run on `attempt-001`, and added direct greedy transformer
answer-training evidence under
`runs/transformer-answer-v0.32-direct-base-context32/`. Direct transformer loss
improved, but strict raw greedy exact answers remained `0/219`.

v0.33 kept corpus sources unchanged from v0.32, passed the archived
self-improvement run on `attempt-001`, and added first-error unlikelihood
transformer evidence under
`runs/transformer-answer-v0.33-unlikelihood-context32/`. Transformer-only
candidate accuracy improved to `37/219`, but strict raw greedy exact answers
remained `0/219`.

v0.34 kept corpus sources unchanged from v0.33, passed the archived
self-improvement run on `attempt-001`, and added staged rollout unlikelihood
transformer evidence under `runs/transformer-answer-v0.34-staged-context32/`.
Direct target loss improved and the repeated-output failure changed shape, but
strict raw greedy exact answers remained `0/219`.

v0.35 kept corpus sources unchanged from v0.34, passed the archived
self-improvement run on `attempt-001`, and added periodic rollout unlikelihood
transformer evidence under
`runs/transformer-answer-v0.35-periodic10-context32/`. Candidate discrimination
returned to `37/219`, but strict raw greedy exact answers remained `0/219`.

v0.36 kept corpus sources unchanged from v0.35, passed the archived
self-improvement run on `attempt-001`, and added periodic early-stop
unlikelihood transformer evidence under
`runs/transformer-answer-v0.36-periodic-earlystop10-context32/`. Candidate
discrimination stayed at `37/219`, answer-target NLL improved to `2.9311`, and
strict raw greedy exact answers remained `0/219` with a repeated `" a"` loop.

v0.37 kept corpus sources unchanged from v0.36, passed the archived
self-improvement run on `attempt-001`, and added periodic repeat-loop
unlikelihood transformer evidence under
`runs/transformer-answer-v0.37-periodic-repeat50-context32/`. Candidate
discrimination stayed at `37/219`, answer-target NLL improved to `2.9041`, and
strict raw greedy exact answers remained `0/219` with a repeated `" t"` loop.

v0.38 kept corpus sources unchanged from v0.37, passed the archived
self-improvement run on `attempt-001`, and added periodic balanced repair
transformer evidence under
`runs/transformer-answer-v0.38-periodic-balanced50-context32/`. Candidate
discrimination stayed at `37/219`, answer-target NLL improved to `2.8552`, and
strict raw greedy exact answers remained `0/219` with a repeated `" t"` loop.

v0.39 kept corpus sources unchanged from v0.38, passed the archived
self-improvement run on `attempt-001`, rejected two generated-prefix recovery
pilots, and added periodic sequence-repair transformer evidence under
`runs/transformer-answer-v0.39-periodic-sequence50-context32/`. Candidate
discrimination stayed at `37/219`, answer-target NLL improved to `2.8257`, and
strict raw greedy exact answers remained `0/219` with a repeated `" t"` loop.
