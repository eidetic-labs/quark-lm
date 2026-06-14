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

v0.40 kept corpus sources unchanged from v0.39, passed the archived
self-improvement run on `attempt-001`, rejected a loop-escape-only pilot, kept a
protected sequence-plus-loop pilot as non-selected evidence, and added branch
repair transformer evidence under `runs/transformer-answer-v0.40-branch-context32/`.
Candidate discrimination stayed at `37/219`, answer-target NLL improved to
`2.5427`, and strict raw greedy exact answers remained `0/219` with a repeated
`"ten"` loop.

v0.41 kept corpus sources unchanged from v0.40, passed the archived
self-improvement run on `attempt-001`, rejected a full-dose branch-contrast
pilot, and added sparse branch-repair/contrast transformer evidence under
`runs/transformer-answer-v0.41-branch-repair-contrast50-context32/`. Candidate
discrimination stayed at `37/219`, answer-target NLL improved to `2.4734`, and
strict raw greedy exact answers remained `0/219` with a repeated `"te"`/`"e"`
loop.

v0.42 kept corpus sources unchanged from v0.41, passed the archived
self-improvement run on `attempt-001`, and widened the sparse branch-contrast
transformer evidence under
`runs/transformer-answer-v0.42-branch-repair-contrast50-dim8-context32/`.
Candidate discrimination stayed at `37/219`, answer-target NLL improved to
`2.4129`, and strict raw greedy exact answers remained `0/219` with the short
wrong completion `" te."`.

Post-v0.42 unpromoted transformer work kept corpus sources unchanged and added
runtime, diagnosis, and rejected checkpoint evidence. The transformer forward
pass now computes only the final position used by the language-model head.
Answer-training artifacts now record prompt context-coverage metrics. The
hard-negative context-32 run
`runs/transformer-answer-v0.43-hard-branch-contrast4-dim8-context32/` preserved
`37/219` candidate discrimination but regressed loss, NLL, and greedy output.
The context-80 run
`runs/transformer-answer-v0.43-branch-repair-contrast50-dim8-context80/`
covered all semantic eval templates (`219/219`) but still trailed v0.42 on
direct loss and answer NLL. The 1500-step context-80 run reached `38/219`
candidates but regressed other promotion metrics. Optional layer normalization
was added as a tested architecture flag, but the context-80 screen
`runs/transformer-answer-v0.43-layernorm-screen-dim8-context80/` preserved only
`37/219` candidates and regressed answer NLL with repeated `" y"`/`"e"` greedy
loops. Branch-span repair was added as a tested direct-answer policy, but
`runs/transformer-answer-v0.43-branch-span3-screen-dim8-context32/` preserved
only `37/219` candidates and regressed answer NLL with a long `"neeee"` greedy
loop. Multi-layer transformer support was added as a tested architecture option,
but `runs/transformer-answer-v0.43-two-layer-screen-dim8-context32/` was
interrupted before final direct-answer metrics because the full-block scalar
autograd path was too slow for the regular loop. The final stacked layer was
then optimized to compute only the last state with logit-equivalence coverage,
but `runs/transformer-answer-v0.43-two-layer-finalopt-screen-dim8-context32/`
was still interrupted before final metrics because intermediate full-state
training remains too expensive. A later bounded screen added top-layer-only
direct-answer updates for stacked transformers plus the explicit
`--skip-post-direct-snapshot` control:
`runs/transformer-answer-v0.43-two-layer-toponly-skip-screen-dim8-context32/`
completed, saved a checkpoint, recorded that the post-direct candidate snapshot
was skipped, improved direct-answer target loss from `3.5186` to `3.2436`, and
still failed direct greedy exact at `0/219` with repeated `"a"` output. It is
runtime and training-loop evidence, not promotion evidence. Direct-answer
snapshots then gained branch-profile diagnostics under
`runs/transformer-answer-v0.43-branch-profile-smoke-dim4-context16/`, recording
the model's own branch-position logits, dominant predicted tokens, target-token
distribution, and target margin. The smoke profile showed QA branch accuracy
staying at `1/8` while dominant prediction moved from all `"o"` to all `"y"`,
which is prompt-independent branch-collapse evidence. Branch-collapse repair
then used the dominant sampled branch token as the unlikelihood negative. The
full-dose smoke at
`runs/transformer-answer-v0.43-branch-collapse-smoke-dim4-context16/` regressed
loss and moved collapse to all `"a"` predictions. The periodic smoke at
`runs/transformer-answer-v0.43-periodic-branch-collapse-smoke-dim4-context16/`
improved direct loss to `3.5157`, but branch accuracy stayed `1/8` and the
dominant prediction moved to all `"n"`. Branch-batch contrast then trained
several distinct target branches in one update. The full-dose smoke at
`runs/transformer-answer-v0.43-branch-batch-smoke-dim4-context16/` improved
loss only slightly and moved collapse to all `"y"` predictions. The periodic
smoke at
`runs/transformer-answer-v0.43-periodic-branch-batch-smoke-dim4-context16/`
improved direct loss to `3.5248`, but QA branch accuracy regressed to `0/8`
and the dominant prediction moved to all `"a"`. A representation-side
context-mean option was then added without changing corpus sources. The
branch-batch screen
`runs/transformer-answer-v0.43-context-mean-branch-batch-smoke-dim4-context16/`
improved direct loss to `3.5252`, and the branch-repair screen
`runs/transformer-answer-v0.43-context-mean-branch-repair-smoke-dim4-context16/`
improved direct loss to `3.5310`; both regressed QA branch accuracy to `0/8`
and collapsed the dominant prediction to all `"a"`. A learned context-projection
option followed, again without changing corpus sources. The branch-repair
screen
`runs/transformer-answer-v0.43-context-projection-branch-repair-smoke-dim4-context16/`
improved direct loss to `3.5217`, and the branch-batch screen
`runs/transformer-answer-v0.43-context-projection-branch-batch-smoke-dim4-context16/`
improved direct loss to `3.5252`; both moved their projection weights, regressed
QA branch accuracy to `0/8`, and collapsed the dominant prediction to all
`"a"`. A prompt-attention summary option followed, again without changing corpus
sources. The branch-repair screen
`runs/transformer-answer-v0.43-prompt-attention-branch-repair-smoke-dim4-context16/`
improved direct loss to `3.5217`, and the branch-batch screen
`runs/transformer-answer-v0.43-prompt-attention-branch-batch-smoke-dim4-context16/`
improved direct loss to `3.5252`; both moved their zero-initialized output
projection weights, regressed QA branch accuracy to `0/8`, and collapsed the
dominant prediction to all `"a"`. Branch-context coverage diagnostics were then
added to direct-answer snapshots without changing corpus sources. The context-16
screen
`runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context16/`
showed QA branch contexts had `0/8` semantic coverage and `4` ambiguous branch
windows. The context-32 screen
`runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context32/`
removed QA ambiguity but still had `0/8` semantic coverage. The context-80
screen
`runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context80/`
reached complete branch-context coverage across all eval sets (`219/219`) with
zero ambiguous branch contexts. The branch-context gate was then added as an
opt-in training guardrail, again without changing corpus sources. The context-16
gate screen
`runs/transformer-answer-v0.43-branch-context-gate-smoke-dim4-context16/`
failed the required gate and recorded `actual_steps: 0` for `5` requested
direct-answer steps. The context-80 gate screen
`runs/transformer-answer-v0.43-branch-context-gate-smoke-dim4-context80/`
passed the required gate and recorded `actual_steps: 1` for `1` requested
direct-answer step. Branch-only direct-answer snapshots were then added as an
explicit screening mode for longer-context repair runs, again without changing
corpus sources. The context-80 gated branch-only screen
`runs/transformer-answer-v0.43-branch-context-gated-branchonly-smoke-dim4-context80/`
passed the required gate across all `219/219` semantic records, ran all `5`
requested direct-answer steps, and recorded skipped greedy evals while retaining
branch profiles and branch-context gate evidence. Two dim8 context-80
branch-only follow-up screens then tested the best prior sparse repair/contrast
policy and branch-batch contrast under complete branch context. The
repair/contrast screen
`runs/transformer-answer-v0.43-branchonly-periodic-repair-contrast50-dim8-context80/`
ran all `100` requested direct steps and lowered interval train loss, but final
QA branch prediction collapsed to all `"a"`. The branch-batch screen
`runs/transformer-answer-v0.43-branchonly-branch-batch-dim8-context80/` ran all
`50` requested direct steps and lowered interval train loss further, but also
collapsed QA branch prediction to all `"a"`. Branch diversity was then promoted
from narrative diagnosis to an explicit snapshot target, again without changing
corpus sources. The smoke run
`runs/transformer-answer-v0.43-branch-diversity-target-smoke-dim4-context80/`
passed the branch-context gate, ran all `5` requested direct steps, and recorded
`branch_diversity_target` failure across all `9` multi-target eval profiles.
The first diversity-aware training objective was then added as
`branch-diversity-unlikelihood`, still without changing corpus sources. The
context-80 smoke
`runs/transformer-answer-v0.43-branch-diversity-train-smoke-dim4-context80/`
passed the branch-context gate and ran all `10` requested direct steps, but the
diversity target still failed across all `9` multi-target eval profiles.
Output-bias freezing was then added as a direct-answer stabilizer, still without
changing corpus sources. The context-80 smoke
`runs/transformer-answer-v0.43-branch-diversity-freezebias-smoke-dim4-context80/`
passed the branch-context gate and ran all `50` requested direct steps with
`--direct-answer-freeze-output-bias`, but the diversity target still failed
across all `9` multi-target eval profiles.
A restricted branch-target softmax objective followed, still without changing
corpus sources. The context-80 smoke
`runs/transformer-answer-v0.43-branch-target-softmax-freezebias-smoke-dim4-context80/`
passed the branch-context gate, froze output bias, and ran all `50` requested
direct steps. It briefly raised QA predicted diversity to two tokens at step
`20`, but the final diversity target still failed across all `9` multi-target
eval profiles.
Best-branch-snapshot restoration followed, still without changing corpus
sources. The context-80 smoke
`runs/transformer-answer-v0.43-branch-target-softmax-restorebest-smoke-dim4-context80/`
passed the branch-context gate, froze output bias, ran all `50` requested direct
steps, and restored the final checkpoint from step `40`; the final diversity
target still failed across all `9` multi-target eval profiles.
Prompt-prefix projection followed, still without changing corpus sources. The
context-80 smoke
`runs/transformer-answer-v0.43-prompt-prefix-target-softmax-restorebest-smoke-dim4-context80/`
passed the branch-context gate, moved all `20` prompt-prefix projection
parameters, and restored the final checkpoint from step `40`; the final
diversity target still failed across all `9` multi-target eval profiles.
Prompt-position projection followed, still without changing corpus sources. The
context-80 smoke
`runs/transformer-answer-v0.43-prompt-position-target-softmax-restorebest-smoke-dim4-context80/`
passed the branch-context gate, moved `1108/1284` prompt-position projection
parameters, and restored the final checkpoint from step `40`; the final
diversity target still failed across all `9` multi-target eval profiles.
A pairwise branch-target margin objective followed, still without changing
corpus sources. The prompt-position context-80 smoke
`runs/transformer-answer-v0.43-branch-target-margin-prompt-position-smoke-dim4-context80/`
passed the branch-context gate, ran all `50` direct steps, moved train loss
`4.8973 -> 4.7784`, moved `1108/1284` prompt-position projection parameters,
and restored the final checkpoint from step `40`; the final diversity target
still failed across all `9` multi-target eval profiles.
Branch representation diagnostics and contrastive hidden-state training
followed, still without changing corpus sources. The high-weight
prompt-position context-80 smoke
`runs/transformer-answer-v0.43-branch-representation-contrast50-prompt-position-smoke-dim4-context80/`
recorded hidden-state distance profiles, ran all `50` direct steps with
`--direct-answer-contrast-weight 50.0`, and restored the final checkpoint from
step `40`; the final diversity target still failed across all `9` multi-target
eval profiles.
A dim-8 capacity screen followed, still without changing corpus sources. The
completed 40-step prompt-position context-80 smoke
`runs/transformer-answer-v0.43-branch-representation-contrast50-prompt-position-smoke-dim8-context80-steps40/`
used embedding/feed-forward dimensions `8/16`, restored the final checkpoint
from step `10`, and increased measured QA hidden distance; the final diversity
target still failed across all `9` multi-target eval profiles.
Prompt-position scale screening followed, still without changing corpus
sources. The scale-32 context-80 smoke
`runs/transformer-answer-v0.43-prompt-position-scale32-repcontrast50-smoke-dim4-context80/`
ran all `50` direct steps, moved `1108/1284` prompt-position projection
parameters, restored the final checkpoint from step `40`, and increased
restored QA hidden distance to about `0.01235`; the final diversity target
still failed across all `9` multi-target eval profiles.
The next checkpoint records an engineering-only open-source structure audit in
`STRUCTURE_AUDIT.md`: QuarkLM may study model/trainer/tokenizer/checkpoint
patterns, but no external weights, tokenizer vocabularies, embeddings,
datasets, or training text enter the corpus or learned artifacts. The audit
selects an opt-in pre-layer-norm transformer block path with final
normalization as the next structural screen before another branch-loss repair.
That path was implemented and screened in
`runs/transformer-answer-v0.44-prelayernorm-repcontrast50-prompt-position-smoke-dim4-context80/`
without changing corpus sources. The run moved prompt-position and final-norm
parameters and cracked full collapse in `7/9` multi-target profiles, but QA and
heldout remained collapsed and the final diversity target still failed across
all `9` multi-target eval profiles.
None of these runs were promoted.
