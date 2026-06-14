# Open-Source Mechanics Audit

Last updated: 2026-06-14.

This audit is the deeper follow-up to `STRUCTURE_AUDIT.md`. The earlier audit
looked at high-level transformer structure. This pass looks at the mechanics
around training loops, replay, checkpoint selection, tokenizer growth,
evaluation, and self-improvement controls.

## Boundary

Allowed:

- Study public project structure, trainer loops, evaluation cadence, replay
  policies, tokenizer interfaces, checkpoint metadata, and transparency
  practices.
- Reimplement lessons in QuarkLM-native code when they advance the
  closed-world goal.
- Use public papers and open-source code as design references.

Not allowed:

- Copying implementation code into QuarkLM.
- Importing pretrained weights, pretrained tokenizers, external embeddings,
  datasets, or unledgered text into training.
- Using an external model as the hidden judge for admission, promotion, or
  repair.
- Treating retrieval, memory, or generated candidates as proof of learned
  parametric behavior.

## Sources Reviewed

| Source | Mechanics studied | QuarkLM lesson |
| --- | --- | --- |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Compact model/trainer split, batch loading, gradient accumulation, AdamW grouping, warmup/cosine scheduling, best validation checkpointing, and generation traces. | QuarkLM should keep the model small, but the trainer should become more explicit: update plans, replay plans, checkpoint scoring, and optimizer state should be artifacts rather than ad hoc branches inside one large file. |
| [minGPT](https://github.com/karpathy/minGPT) | Educational separation of model, trainer, dataset, and callback-style progress hooks. | QuarkLM can preserve scalar-audited clarity while still separating direct-answer batch construction, objective computation, and snapshot scoring. |
| [LitGPT](https://github.com/Lightning-AI/litgpt) | Config-driven decoder-only model variants, rotary positions, norm variants, KV-cache behavior, recipe separation, and memory-aware logits handling. | v0.51 added many of the right primitives, but QuarkLM still needs a clearer config and recipe boundary before adding many more branch-objective modes. |
| [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) | Tokenizer pipeline stages: normalization, pre-tokenization, model/trainer, special tokens, padding, truncation, decoding, alignment, and serialization. | Character tokenization remains the purity baseline, but any future subword path needs a manifest, trainer artifact, offsets, special-token policy, and admitted-corpus-only vocabulary proof. |
| [Avalanche](https://github.com/ContinualAI/avalanche) | Continual-learning streams, strategies, replay plugins, generative replay, evaluation plugins, logging, and baseline comparisons. | Replay should be a first-class QuarkLM primitive with source/profile coverage, not just an extra term in a branch loss. Retention and forgetting metrics need to be attached to every staged update. |
| [Deep Generative Replay](https://arxiv.org/abs/1705.08690) | Solver/generator separation for replaying prior tasks without storing every original sample. | QuarkLM should prefer replay of admitted originals first. Generated replay is only admissible when it is reconstructable from admitted sources and verifier-approved. |
| [InsCL](https://arxiv.org/abs/2403.11435) | Instruction-aware replay selection based on task similarity and instruction information. | The closest QuarkLM analogue is profile-aware replay: `qa:place`, `qa:color`, heldout, glossary, learning, and admission profiles should not share one global coverage target. |
| [Self-Instruct](https://arxiv.org/abs/2212.10560) | Generate candidate instruction data, filter invalid or duplicate items, then train on accepted samples. | QuarkLM's "I learned something new" loop should generate candidates but admit only deterministic or internally verified lessons. |
| [STaR](https://arxiv.org/abs/2203.14465) | Iterative self-generated rationales filtered by answer correctness before fine-tuning. | Self-generated repair traces can become training data only after closed-world verification; correctness gates matter more than volume. |
| [Reflexion](https://arxiv.org/abs/2303.11366) | Trial feedback stored as linguistic memory without weight updates. | QuarkLM should continue separating memory artifacts, exact responders, retrieval-like rails, and weight learning claims. |
| [LLM360](https://arxiv.org/abs/2312.06550) | Transparency standard: code, data, checkpoints, intermediate results, and analyses released together. | QuarkLM's run artifacts, docs, failed screens, and checkpoints are not overhead; they are part of the research claim. |
| [OLMo](https://arxiv.org/abs/2402.00838) and [OLMo 2](https://arxiv.org/abs/2501.00656) | Open training/evaluation code, training data, recipes, logs, intermediate checkpoints, data mixtures, and late-stage curriculum. | QuarkLM should publish the recipe and rejected evidence for each version, and should report corpus/source mixtures before training rather than only final metrics after training. |

## Findings

### 1. The Current Bottleneck Is Trainer Mechanics

QuarkLM has already implemented a serious from-scratch transformer foundation:
AdamW-style optimizer state, scheduling, gradient accumulation, RMSNorm, gated
MLPs, tied embeddings, rotary positions, checkpoint metadata, eval samples, and
branch diagnostics. The failure pattern after v0.65 is no longer "missing
standard transformer pieces." It is that direct-answer objectives are fighting
each other without a sufficiently explicit replay and promotion plan.

The next useful improvement is not another global loss term. It is a trainer
mechanics improvement: construct profile-aware replay plans, train deficits per
profile, preserve represented coverage per profile, and record the plan as an
artifact.

### 2. Replay Needs Profiles, Not One Global Target Set

v0.61 through v0.65 repeatedly showed the same shape: rank improves, but
coverage collapses or one represented target is over-preserved. The audit
points to the reason: mature continual-learning systems treat task streams,
experience identity, and evaluation metrics as first-class. QuarkLM's branch
repair loop currently lets all branch targets compete in a mostly global pool.

For QuarkLM, the right unit is a closed-world profile:

- eval set or source family, such as QA, heldout, admission, learning, self, or
  glossary;
- lesson subtype, such as place, color, owner, relation, or unknown policy;
- branch position and context-coverage state;
- original versus generated-candidate status.

Replay should preserve coverage inside each profile before a snapshot can be
eligible for promotion.

### 3. Checkpoint Selection Must Be Constraint First

nanoGPT-style "best validation loss" is too broad for QuarkLM's direct-answer
screens. v0.60 moved in the right direction by adding a target-token coverage
floor before rank/top-k scoring. The audit suggests tightening that discipline:
checkpoint scoring should first enforce profile coverage, unknown-policy,
leakage, and retention constraints, then rank candidate snapshots.

That means "best" is not the lowest loss. Best is the checkpoint that preserves
the closed-world invariants while making the targeted behavior more true.

### 4. Self-Improvement Should Produce Candidate Artifacts First

Self-Instruct, STaR, and Reflexion all separate generation from acceptance in
different ways. For QuarkLM, that maps cleanly to:

1. propose a candidate lesson, probe, replay item, or repair note;
2. verify it against admitted sources and current eval policy;
3. admit it with provenance if it passes;
4. train weights from the admitted curriculum;
5. report whether the weight update improved, preserved, or regressed behavior.

This keeps future self-improvement possible without an external model shaping
the learner.

### 5. Tokenizer Growth Is Real, But Still Deferred

The tokenizer sources show that tokenizer growth is an artifacted training
problem, not a helper function. A subword tokenizer would need a corpus-only
trainer, special-token policy, serialization, offsets/alignment, migration
tests, checkpoint compatibility, and eval comparability. The current evidence
still points more strongly at prompt-to-token binding and replay mechanics than
at character-token length as the limiting factor.

## QuarkLM Gap Matrix

| Area | Current state | Gap | Required next move |
| --- | --- | --- | --- |
| Trainer boundary | `transformer_char_model.py` owns model, CLI, branch batches, direct objectives, snapshot metrics, and run writing. | More modes make the file harder to reason about and easier to regress. | Extract direct-answer batch planning, objective selection, snapshot scoring, and replay reporting behind narrow functions or small classes. |
| Replay plan | Replay targets are computed inside objective functions. | No explicit artifact proves which profiles, targets, deficits, and represented tokens were trained. | Write a replay-plan artifact for direct-answer screens with per-profile target counts, represented targets, deficits, branch counts, and coverage floors. |
| Profile identity | `AnswerExample.source` exists, but many objectives collapse targets into one global set. | QA, heldout, admission, glossary, learning, and self profiles can damage each other. | Carry profile keys through branch records and compute deficits/preservation inside each profile. |
| Checkpoint scoring | v0.60 added a profile-wise target-token coverage floor. | The floor is not yet tied to the trainer's replay plan, and some objectives still preserve the wrong represented token. | Make coverage preservation profile-aware and require coverage evidence before rank/top-k evidence is considered. |
| Data mixture | Corpus admission is ledgered, but direct training screens do not always expose mixture balance. | A screen can lower loss by over-sampling easy or common profiles. | Report source/profile mixture for every direct-answer training plan. |
| Candidate lessons | Self-diagnosis is deterministic and candidate generation remains mostly human-driven. | The model cannot yet autonomously propose and verify new training material. | Add candidate lesson/probe artifacts that are excluded from training until closed-world verification accepts them. |
| Tokenizer | Character tokenizer is corpus-only and strict. | Future subword migration would be easy to contaminate without artifacts. | Keep character baseline; design any subword trainer as a separate admitted-corpus-only artifact with offsets and manifest checks. |
| Evidence discipline | Failed transformer screens are documented in README/status/current-state. | GOAL and docs can still lag behind latest mechanics unless each version updates shared state. | Treat mechanics audits and rejected runs as versioned evidence with README, Docusaurus, HISTORY, STATUS, and current-state updates. |

## Implementation Requirements Before The Next Full-Stack Repair Run

1. Branch replay records must carry a profile key derived from admitted
   example metadata.
2. Coverage deficits must be computed per profile, not globally.
3. Represented-target preservation must preserve each profile's coverage
   without over-anchoring one global represented token.
4. The direct-answer training loop should emit a replay-plan artifact that
   records branch counts, replay counts, target sets, represented targets,
   missing targets, and coverage floors by profile.
5. Focused tests must cover profile isolation: improving one profile must not
   satisfy or mask another profile's coverage deficit.
6. Snapshot scoring must keep the v0.60 coverage-first discipline and compare
   rank/top-k evidence only after coverage constraints pass.
7. Documentation must state whether a version is model-quality evidence,
   mechanics-readiness evidence, audit evidence, or promotion evidence.

## Decision

v0.66 should be treated as an open-source mechanics audit and gap-setting
version. The next implementation should continue the in-progress
profile-aware coverage-constrained repair, but only with explicit replay-plan
artifacts and profile-isolation tests. That moves the project forward without
copying outside code or diluting the closed-world training boundary.
