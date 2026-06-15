---
title: Transformer
description: Train the from-scratch QuarkLM transformer prototype.
---

# Transformer

v0.24 introduced a tiny decoder-only transformer in
`closed_world_lm.transformer_char_model`.

It is intentionally small:

- corpus-trained character tokenizer
- learned token and position embeddings
- one causal self-attention block
- one feed-forward block
- next-character language-model head
- dependency-free scalar autodiff
- random initialization only

Train a smoke checkpoint:

```bash
PYTHONPATH=src python3 -m closed_world_lm.transformer_char_model train \
  --run runs/transformer-smoke \
  --steps 40 \
  --context-size 8 \
  --embedding-dim 6 \
  --feedforward-dim 12
```

Evaluate answer probes:

```bash
PYTHONPATH=src python3 -m closed_world_lm.transformer_char_model eval \
  --checkpoint runs/transformer-smoke/transformer.json \
  --json runs/transformer-smoke/transformer_eval.json
```

Train on corpus-derived answer lessons:

```bash
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

From v0.71 onward, `answer-train` writes `experiment_intent.json` before
training and closes it with a decision in `transformer_answer_metrics.json`.
Use `--experiment-hypothesis`, `--experiment-acceptance-gate name:rule`,
`--experiment-failure-criterion`, and `--experiment-note` to make a screen's
intent more specific. From v0.77 onward, transformer screens close through the
constraint-first promotion report.

From v0.72 onward, profile-aware replay planning lives in
`src/closed_world_lm/replay_plan.py`. The transformer still emits the same
`direct_answer_replay_plan.json` shape for profile-aware modes, but replay
record normalization, profile grouping, coverage floors, and missing-target
summaries are now standalone training-planning mechanics.

From v0.73 onward, `answer-train` also writes `corpus_hygiene.json` and
`training_plan.json`. These artifacts record source mixture, duplicate checks,
train/eval prompt overlap, candidate ratio, rare-profile coverage, allowed data
sources, planned artifacts, and replay-plan summaries when profile-aware replay
writes a plan.

From v0.75 onward, `answer-train` also writes
`candidate_quarantine.json`. The manifest records candidate lifecycle state and
is linked from `training_plan.json`; candidate records are not training data
until admitted into the ledgered corpus.

From v0.76 onward, `answer-train` also writes
`closed_world_verifier.json`. The verifier is deterministic and checks that the
closed-world data boundary, candidate exclusion policy, quarantine manifest, and
protected train/eval overlap all pass before transformer screen evidence is
trusted.

From v0.77 onward, `answer-train` also writes `training_recipe.json` and
`constraint_first_promotion.json`. The recipe records model, tokenizer, data,
objective, optimizer, replay, artifacts, gates, and rerun details. The
constraint-first report blocks loss, NLL, rank, top-k, or exact quality
evidence until verifier, contamination, branch-context, coverage, and diversity
constraints pass first.

From v0.78 onward, the answer-training stack starts using separate
[transformer responsibility surfaces](./transformer-responsibilities.md) for
artifact contracts, experiment/recipe decisions, trainer utilities, and the
direct-answer objective catalog. The public CLI and artifact names remain
stable.

From v0.79 onward, `src/closed_world_lm/transformer_model.py` owns model,
optimizer, and generation config validation, checkpoint identity, closed-world
dataset metadata, and run metadata. `transformer_char_model.py` still exports
the old names for compatibility.

From v0.80 onward, `src/closed_world_lm/transformer_checkpoint.py` owns
checkpoint payload loading and identity validation, and
`src/closed_world_lm/transformer_eval.py` owns generic transformer probe
loading, candidate collection, scoring, eval report assembly, samples JSONL
writing, and eval JSON writing. The public `eval` CLI and artifact shapes
remain stable.

From v0.81 onward,
`branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood`
adds balanced owned target-share pressure across replay targets inside each
profile-aware replay group. It keeps the existing profile replay plan,
deficit focus, and represented-target preservation, but adds a per-target
anti-collapse term so one represented target cannot dominate a multi-target
profile without pressure on the remaining replay targets.

v0.82 screens that objective under the modern artifact stack and
constraint-first gates. The screen fixes the transformer metrics purity field
for `external_embeddings`, passes the verifier and branch-context gate, and
preserves coverage by restoring step `0`, but trained snapshots still collapse
QA and heldout branch diversity.

v0.83 adds
`branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
It keeps the profile target-share objective and adds a prompt-specific
sibling-target margin, so each replay context is trained to rank its own target
above other targets from the same profile. The focused mechanic passes, but the
full screen still rejects trained snapshots that lose target-token coverage.

v0.84 adds
`branch-balanced-context-profile-baseline-anchored-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
It keeps prompt ownership but anchors replay preservation to the baseline replay
predictions recorded before direct-answer training, so preservation no longer
follows prediction drift. The screen improves trained coverage relative to
v0.83 but still restores baseline because it misses the full coverage floor.

v0.85 adds
`branch-balanced-context-profile-baseline-floor-gated-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
It keeps baseline replay anchors and rejects any attempted direct-answer update
whose branch-profile target-token coverage falls below the step-0 floor. The
screen preserves coverage by rejecting all attempted updates, so the next repair
must produce accepted safe updates rather than looser promotion gates.

v0.86 adds
`branch-balanced-context-profile-baseline-floor-adaptive-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
It keeps the baseline-floor guard and retries the same update at smaller
learning-rate scales after restoring model, optimizer, and RNG state. The screen
shows that step size alone is not enough: all scaled attempts are still rejected.

v0.87 adds
`branch-balanced-context-profile-baseline-floor-repaired-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
It keeps the adaptive guard and adds one bounded baseline-covered anchor repair
before each failed retry is accepted or rejected. The screen shows that
post-update repair is not enough: all repaired attempts are still rejected.

v0.88 adds
`branch-balanced-context-profile-baseline-floor-objective-prompt-ownership-target-share-preserving-deficit-unlikelihood`.
It puts balanced baseline-floor anchors inside the same loss and backward pass
as the branch-diversity pressure. The screen shows that the combined objective
is still not enough: all objective-shaped attempts are rejected.

v0.89 adds `branch-context-profile-baseline-floor-stabilization-unlikelihood`.
It removes branch-diversity pressure from guarded attempts and trains only
baseline-covered floor anchors. The screen shows that floor-only stabilization
is still not enough: all stabilization-shaped attempts are rejected.

v0.90 adds baseline-floor rejection diagnostics to the same stabilization mode.
The guard now records rejected update-shape counts, rejected learning-rate scale
counts, violation profile counts, compact per-attempt floor diagnostics, and the
worst rejected coverage violation.

v0.91 adds
`branch-context-profile-baseline-floor-profile-targeted-stabilization-unlikelihood`.
It covers every baseline-covered floor-anchor profile-target group in each
guarded attempt. The screen shows that broader floor-anchor coverage is still
not enough: all profile-targeted attempts are rejected with the same violation
pattern as v0.90.

v0.92 adds
`branch-context-profile-baseline-floor-sequential-profile-stabilization-unlikelihood`.
It changes the repair shape to sequential source-profile floor batches with
rollback after each unsafe profile group. The screen shows that source-profile
ordering is still not enough: all profile-local attempts are rejected before any
effective guarded update survives.

v0.93 adds
`branch-context-profile-baseline-floor-calibrated-sequential-profile-stabilization-unlikelihood`.
It keeps the sequential rollback shape, extends calibrated adaptive scales below
`0.01`, and uses coverage-only guard probes. The diagnostic screen accepts the
first nonzero source-profile update that preserves the baseline floor.

v0.94 adds
`branch-context-profile-baseline-floor-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood`.
It searches calibrated scales separately for each source profile, keeps the
first safe update for that profile, and rolls back only unsafe profile-scale
attempts. The diagnostic screen accepts eight source-profile updates while the
baseline floor remains preserved.

v0.95 adds
`branch-context-profile-baseline-floor-diversity-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood`.
It keeps the profile-scale search, then accepts a source-profile update only
when it preserves the baseline coverage floor and does not regress the
branch-diversity score from that profile's pre-update state. The diagnostic
screen accepts five score-improving source-profile updates and rejects eleven
floor-preserving score regressions before promotion still blocks on branch
diversity.

Add `--use-context-mean` to either `train` or `answer-train` to test the
experimental mean-pooled context residual in the final transformer
representation. It is diagnostic architecture evidence only until it improves
prompt-conditioned branch profiles and complete answer metrics.
Add `--use-context-projection` to test a zero-initialized trainable projection
of that context summary; it starts baseline-equivalent and must prove that its
learned parameters improve branch profiles before it can be promoted.
Add `--use-prompt-prefix-projection` to test a zero-initialized trainable
projection of non-padding prompt-prefix positions before the final answer token.
Add `--use-prompt-attention-summary` to test a trainable attention-pooled
summary of the current context through a zero-initialized output projection.
It is also diagnostic until branch profiles improve.
Add `--direct-answer-require-branch-context-gate` to skip direct-answer
training unless branch contexts are semantically complete and unambiguous.
Add `--direct-answer-snapshot-mode branch-only` for bounded longer-context
screens that need branch profiles and branch-context gate evidence but can
intentionally skip greedy completion evals in direct-answer JSONL snapshots.
Direct-answer snapshots also emit `branch_diversity_target`, which fails when
multi-target eval profiles collapse to too few predicted branch tokens.
Use `--direct-answer-mode branch-diversity-unlikelihood` to train distinct
branch targets while also suppressing each branch context's current wrong
prediction.
Use `--direct-answer-freeze-output-bias` to exclude the transformer output bias
from direct-answer updates when screening whether a branch objective is learning
prompt-specific weights rather than moving one global token bias.
Use `--direct-answer-mode branch-target-softmax-unlikelihood` to add a
restricted softmax over the distinct branch targets in each batch, making the
right target compete directly against the other observed branch targets.
Use `--direct-answer-restore-best-branch-snapshot` to restore the best scored
branch-diversity checkpoint before final metrics and checkpoint writing.
Add `--use-prompt-position-projection` to test a zero-initialized
position-specific projection of non-padding prompt-prefix positions.
Add `--prompt-position-projection-scale` to scale that prompt-position
projection residual before it is added to the final branch representation.
Use `--direct-answer-mode branch-target-margin-unlikelihood` to add a smooth
pairwise target-margin loss over the distinct branch targets in each batch.
Direct-answer snapshots include `branch_representation_profiles` so runs can
measure hidden-state pairwise distance before the output head.
Use `--direct-answer-mode branch-representation-contrast-unlikelihood` to
penalize nearly identical hidden states for different branch targets.
Use `--direct-answer-mode branch-balanced-representation-contrast-unlikelihood`
to build that representation-contrast batch from target buckets so frequent
first answer tokens cannot crowd out rare branch targets.
Direct-answer branch profiles also include target-rank diagnostics: average
target rank, top-3/top-5 target coverage, and the top predicted alternatives on
failed branch records.
Use `--direct-answer-mode branch-output-binding-unlikelihood` to combine
restricted branch-target softmax with branch representation contrast in the
same update.
Use `--direct-answer-mode branch-rank-margin-unlikelihood` to push each branch
target above the model's current top wrong tokens. The
`--direct-answer-hard-negatives` value controls how many top wrong tokens each
branch target is margined against.
Use `--direct-answer-mode branch-balanced-rank-margin-unlikelihood` to apply
the same rank-margin repair with target-balanced branch batches.
Use `--direct-answer-mode branch-topk-softmax-unlikelihood` to train each
branch target against a restricted softmax over the target and the model's
current top wrong tokens. Use
`--direct-answer-mode branch-balanced-topk-softmax-unlikelihood` for the same
objective with target-balanced branch batches. The
`--direct-answer-hard-negatives` value controls the top-wrong-token candidate
count, and `--direct-answer-contrast-weight` controls the restricted-softmax
loss weight.
Use `--direct-answer-mode branch-bidirectional-binding-unlikelihood` to bind
prompt contexts and branch targets in both directions: row-wise target choice
inside each prompt context, and column-wise target-token ownership across
prompt contexts. Use
`--direct-answer-mode branch-balanced-bidirectional-binding-unlikelihood` for
the same objective with target-balanced branch batches.
Use `--direct-answer-mode branch-coverage-binding-unlikelihood` to combine
bidirectional binding with hard-wrong-token competition and a target-set mass
coverage guard. Use
`--direct-answer-mode branch-balanced-coverage-binding-unlikelihood` for the
same objective with target-balanced branch batches, and use
`--direct-answer-hard-negatives` to choose the hard wrong-token pool size.
Use `--direct-answer-mode branch-target-set-coverage-unlikelihood` to train
only target-set mass against hard wrong tokens before exact-target sharpening.
Use `--direct-answer-mode branch-balanced-target-set-coverage-unlikelihood`
for the same objective with target-balanced branch batches.
Use `--direct-answer-mode branch-target-diversity-unlikelihood` to keep
target-set mass pressure while adding an explicit target-share diversity term
over the branch target set. Use
`--direct-answer-mode branch-balanced-target-diversity-unlikelihood` for the
same objective with target-balanced branch batches.
Use `--direct-answer-mode branch-target-replay-coverage-unlikelihood` to apply
target-set mass and target-share balance over the broader admitted branch
training pool at the same branch position. Use
`--direct-answer-mode branch-balanced-target-replay-coverage-unlikelihood` for
the same objective with target-balanced sampled branch batches.
Use `--direct-answer-mode branch-context-replay-coverage-unlikelihood` to train
each sampled replay branch context to own its own target within the replay
target set. Use
`--direct-answer-mode branch-balanced-context-replay-coverage-unlikelihood` for
the same objective with target-balanced sampled branch and replay batches.
Use `--direct-answer-mode branch-context-coverage-anchor-unlikelihood` to add a
covered-target anchor for replay branches whose own target is already top-1.
Use `--direct-answer-mode branch-balanced-context-coverage-anchor-unlikelihood`
for the same objective with target-balanced sampled branch and replay batches.
Use `--direct-answer-mode branch-context-target-balanced-anchor-unlikelihood`
to average covered-target anchors by covered target and skip singleton covered
target batches. Use
`--direct-answer-mode branch-balanced-context-target-balanced-anchor-unlikelihood`
for the same objective with target-balanced sampled branch and replay batches.
Use `--direct-answer-mode branch-context-coverage-deficit-unlikelihood` to
identify replay target tokens that are absent from current replay predictions
and focus extra target pressure on those missing targets. Use
`--direct-answer-mode branch-balanced-context-coverage-deficit-unlikelihood`
for the same objective with target-balanced sampled branch and replay batches.
Use `--direct-answer-mode branch-context-coverage-preserving-deficit-unlikelihood`
to combine missing-target pressure with target-balanced preservation anchors
for target tokens currently represented in replay predictions. Use
`--direct-answer-mode branch-balanced-context-coverage-preserving-deficit-unlikelihood`
for the same objective with target-balanced sampled branch and replay batches.
Use `--direct-answer-mode branch-context-profile-coverage-preserving-deficit-unlikelihood`
to compute those deficits and preservation anchors inside each admitted
source/profile instead of one global replay target set. Use
`--direct-answer-mode branch-balanced-context-profile-coverage-preserving-deficit-unlikelihood`
for the same objective with target-balanced sampled branch and replay batches.
Profile-aware modes emit `direct_answer_replay_plan.json` with branch counts,
replay counts, target ids, represented target ids, missing target ids, and
coverage floors by profile before direct-answer training starts.
Best branch snapshot scoring first enforces a profile-wise target-token
coverage floor against the baseline snapshot. Eligible snapshots then use
target-rank/top-k evidence before generic wrong-token diversity, so restore
prefers snapshots that move correct targets upward without trading away
coverage.
v0.51 adds opt-in foundation-stack controls before the next repair objective:
`--optimizer adamw`, `--gradient-accumulation-steps`, warmup/decay schedule
flags, `--resume-checkpoint`, `--resume-optimizer`, `--attention-heads`,
`--use-rms-norm`, `--use-gated-mlp`, `--tie-output-embeddings`,
`--use-rotary-positions`, `--use-kv-cache-path`, generation sampling controls,
and eval `--samples-jsonl` trace artifacts.
Use `STRUCTURE_AUDIT.md` before adding the next transformer repair objective:
QuarkLM may study open-source model/trainer/tokenizer/checkpoint structure, but
must not import external weights, tokenizers, embeddings, datasets, or training
text. Use `--use-pre-layer-norm` to run the audited opt-in GPT-style
pre-layer-norm block path with final normalization before the language-model
head.

Current language-model evidence from `runs/transformer-v0.25/`:

| Signal | Value |
| --- | --- |
| Steps | `40` |
| Validation NLL | `3.5885 -> 3.4382` |
| Answer exact eval | `0/28` |
| Pretrained weights | `false` |
| Pretrained tokenizer | `false` |

Current promoted answer-lesson evidence from
`runs/transformer-answer-v0.42-branch-repair-contrast50-dim8-context32/`:

| Signal | Value |
| --- | --- |
| Steps | `80` |
| Context size | `32` |
| Embedding / feed-forward dimensions | `8 / 16` |
| Candidate scope | `eval` |
| Direct answer steps | `1000` |
| Direct answer mode | `periodic-branch-repair-contrast-unlikelihood` |
| Direct answer negative weight | `1.0` |
| Direct answer positive weight | `1.0` |
| Direct answer contrast weight | `1.0` |
| Direct answer branch position | `1` |
| Direct answer rollout interval | `50` |
| Direct answer training examples | `9144` |
| Direct answer exact | `0/219 -> 0/219` |
| Direct answer target loss | `3.4278 -> 2.2708` |
| Direct answer uses candidates | `false` |
| Direct answer auxiliary weights | `false` |
| Answer target NLL | `3.5850 -> 2.4129` |
| Transformer-only candidate accuracy | `15/219 -> 37/219` |
| Selector-emitted exact answers | `18/219 -> 219/219` |
| Selector candidate accuracy | `18/219 -> 219/219` |
| v0.31 generator exact without candidates | `0/219 -> 219/219` |
| v0.31 generator target loss | `3.3160 -> 0.0029` |
| Pretrained weights | `false` |
| Pretrained tokenizer | `false` |
| External embeddings | `false` |
| v0.31 generator uses answer candidates | `false` |

Latest bounded stacked-transformer screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.43-two-layer-toponly-skip-screen-dim8-context32/` |
| Layers | `2` |
| Steps | `40` target-loss + `80` direct-answer |
| Direct-answer update scope | top layer and language-model head only |
| Post-direct candidate snapshot | skipped and recorded in metrics |
| Pre-direct candidate accuracy | `15/219 -> 15/219` |
| Pre-direct answer target NLL | `3.5855 -> 3.4796` |
| Direct answer target loss | `3.5186 -> 3.2436` |
| Direct answer exact | `0/219 -> 0/219` |
| Failure pattern | repeated `"a"` greedy completion |
| Promotion status | screening evidence only |

Latest direct-answer diagnostic smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.43-branch-profile-smoke-dim4-context16/` |
| Diagnostic | branch profiles from model logits |
| Branch position | `1` |
| Smoke steps | `5` target-loss + `5` direct-answer |
| Post-direct candidate snapshot | skipped and recorded in metrics |
| QA branch accuracy | `1/8 -> 1/8` |
| Dominant QA branch prediction | all `"o"` -> all `"y"` |
| Final QA target margin | negative, about `-0.0048` |
| Promotion status | diagnostic smoke only |

Latest branch repair smoke:

| Signal | Value |
| --- | --- |
| Selected comparison run | `runs/transformer-answer-v0.43-periodic-branch-batch-smoke-dim4-context16/` |
| Prior rejected repair | `runs/transformer-answer-v0.43-periodic-branch-collapse-smoke-dim4-context16/` |
| Mode | `periodic-branch-batch-contrast-unlikelihood` |
| Branch batch size | `4` |
| Rollout interval | `5` |
| Steps | `5` target-loss + `20` direct-answer |
| Direct answer loss | `3.5800 -> 3.5248` |
| QA branch accuracy | `1/8 -> 0/8` |
| Dominant QA branch prediction | all `"o"` -> all `"a"` |
| Promotion status | rejected repair evidence |

Latest representation-side smoke:

| Signal | Value |
| --- | --- |
| Selected run | `runs/transformer-answer-v0.43-context-mean-branch-repair-smoke-dim4-context16/` |
| Comparison run | `runs/transformer-answer-v0.43-context-mean-branch-batch-smoke-dim4-context16/` |
| Representation option | `--use-context-mean` |
| Selected mode | `periodic-branch-repair-unlikelihood` |
| Comparison mode | `periodic-branch-batch-contrast-unlikelihood` |
| Steps | `5` target-loss + `20` direct-answer |
| Post-direct candidate snapshot | skipped and recorded in metrics |
| Selected direct answer loss | `3.5805 -> 3.5310` |
| Comparison direct answer loss | `3.5805 -> 3.5252` |
| Selected QA branch accuracy | `1/8 -> 0/8` |
| Comparison QA branch accuracy | `1/8 -> 0/8` |
| Dominant QA branch prediction | all `"o"` -> all `"a"` in both screens |
| Promotion status | rejected representation evidence |

Latest learned-representation smoke:

| Signal | Value |
| --- | --- |
| Selected run | `runs/transformer-answer-v0.43-context-projection-branch-repair-smoke-dim4-context16/` |
| Comparison run | `runs/transformer-answer-v0.43-context-projection-branch-batch-smoke-dim4-context16/` |
| Representation option | `--use-context-projection` |
| Selected mode | `periodic-branch-repair-unlikelihood` |
| Comparison mode | `periodic-branch-batch-contrast-unlikelihood` |
| Steps | `5` target-loss + `20` direct-answer |
| Post-direct candidate snapshot | skipped and recorded in metrics |
| Projection parameter movement | all `20` parameters moved in both screens |
| Selected direct answer loss | `3.5802 -> 3.5217` |
| Comparison direct answer loss | `3.5802 -> 3.5252` |
| Selected QA branch accuracy | `1/8 -> 0/8` |
| Comparison QA branch accuracy | `1/8 -> 0/8` |
| Dominant QA branch prediction | all `"o"` -> all `"a"` in both screens |
| Promotion status | rejected representation evidence |

Latest prompt-attention representation smoke:

| Signal | Value |
| --- | --- |
| Selected run | `runs/transformer-answer-v0.43-prompt-attention-branch-repair-smoke-dim4-context16/` |
| Comparison run | `runs/transformer-answer-v0.43-prompt-attention-branch-batch-smoke-dim4-context16/` |
| Representation option | `--use-prompt-attention-summary` |
| Selected mode | `periodic-branch-repair-unlikelihood` |
| Comparison mode | `periodic-branch-batch-contrast-unlikelihood` |
| Steps | `5` target-loss + `20` direct-answer |
| Post-direct candidate snapshot | skipped and recorded in metrics |
| Output projection movement | all `20` zero-initialized parameters moved in both screens |
| Selected direct answer loss | `3.5802 -> 3.5217` |
| Comparison direct answer loss | `3.5802 -> 3.5252` |
| Selected QA branch accuracy | `1/8 -> 0/8` |
| Comparison QA branch accuracy | `1/8 -> 0/8` |
| Dominant QA branch prediction | all `"o"` -> all `"a"` in both screens |
| Promotion status | rejected representation evidence |

Latest branch-context coverage diagnostic:

| Signal | Context 16 | Context 32 | Context 80 |
| --- | --- | --- | --- |
| Run | `runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context16/` | `runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context32/` | `runs/transformer-answer-v0.43-branch-context-coverage-smoke-dim4-context80/` |
| QA semantic coverage | `0/8` | `0/8` | `8/8` |
| QA ambiguous branch contexts | `4` | `0` | `0` |
| All-eval semantic coverage | `0/219` | `53/219` | `219/219` |
| All-eval ambiguous branch contexts | `40` | `0` | `0` |
| Promotion status | diagnostic only | diagnostic only | diagnostic only |

Latest branch-context gate smoke:

| Signal | Context 16 | Context 80 |
| --- | --- | --- |
| Run | `runs/transformer-answer-v0.43-branch-context-gate-smoke-dim4-context16/` | `runs/transformer-answer-v0.43-branch-context-gate-smoke-dim4-context80/` |
| Required gate | `true` | `true` |
| Gate status | failed | passed |
| Requested direct steps | `5` | `1` |
| Actual direct steps | `0` | `1` |
| Training skipped | `true` | `false` |
| Promotion status | guardrail evidence only | guardrail evidence only |

Latest branch-only snapshot smoke:

| Signal | Initial smoke | Repair/contrast screen | Branch-batch screen |
| --- | --- | --- | --- |
| Run | `runs/transformer-answer-v0.43-branch-context-gated-branchonly-smoke-dim4-context80/` | `runs/transformer-answer-v0.43-branchonly-periodic-repair-contrast50-dim8-context80/` | `runs/transformer-answer-v0.43-branchonly-branch-batch-dim8-context80/` |
| Context size | `80` | `80` | `80` |
| Embedding/feed-forward dim | `4/8` | `8/16` | `8/16` |
| Snapshot mode | `branch-only` | `branch-only` | `branch-only` |
| Required gate | passed, `219/219` semantic records covered | passed, `219/219` semantic records covered | passed, `219/219` semantic records covered |
| Requested/actual direct steps | `5/5` | `100/100` | `50/50` |
| JSONL greedy evals skipped | `true` | `true` | `true` |
| QA branch profile | all `"x"` to all `"r"`; `1/8` final | all space to all `"a"`; `0/8` final | all space to all `"a"`; `0/8` final |
| Direct loss signal | smoke only | interval train loss `6.7890 -> 6.4326` | interval train loss `3.4614 -> 3.1976` |
| Promotion status | screening efficiency evidence only | rejected screening evidence | rejected screening evidence |

Latest branch-diversity target smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.43-branch-diversity-target-smoke-dim4-context80/` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `5/5` |
| Snapshot mode | `branch-only` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | `"r"` at rate `1.0` |
| Final QA target-token coverage | `0.125` |
| Promotion status | explicit target evidence only |

Latest branch-diversity training smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.43-branch-diversity-train-smoke-dim4-context80/` |
| Mode | `branch-diversity-unlikelihood` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `10/10` |
| Snapshot mode | `branch-only` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | all `"b"` |
| Final QA target-token coverage | `0.125` |
| Promotion status | rejected training-mode evidence |

Latest branch-diversity freeze-bias smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.43-branch-diversity-freezebias-smoke-dim4-context80/` |
| Mode | `branch-diversity-unlikelihood` |
| Stabilizer | `--direct-answer-freeze-output-bias` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `50/50` |
| Snapshot mode | `branch-only` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Direct answer train loss | `3.6149 -> 3.5016` |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | all `"w"` |
| Final QA target-token coverage | `0.0` |
| Promotion status | rejected stabilizer evidence |

Latest branch-target softmax smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.43-branch-target-softmax-freezebias-smoke-dim4-context80/` |
| Mode | `branch-target-softmax-unlikelihood` |
| Stabilizer | `--direct-answer-freeze-output-bias` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `50/50` |
| Snapshot mode | `branch-only` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Composite train loss | `5.6671 -> 5.5820` |
| Best QA predicted unique | `2/8` at step `20` |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | all `"w"` |
| Final QA target-token coverage | `0.0` |
| Promotion status | rejected target-set evidence |

Latest branch restore-best smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.43-branch-target-softmax-restorebest-smoke-dim4-context80/` |
| Mode | `branch-target-softmax-unlikelihood` |
| Stabilizers | `--direct-answer-freeze-output-bias`, `--direct-answer-restore-best-branch-snapshot` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `50/50` |
| Restored best branch snapshot | yes, from step `40` |
| Best branch score | `[0.0, 0.0, -9.0, 0.0, 0.0946, 0.1409, 0.0]` |
| Snapshot mode | `branch-only` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | all `"u"` |
| Final QA target-token coverage | `0.125` |
| Promotion status | rejected guardrail evidence |

Latest prompt-prefix projection smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.43-prompt-prefix-target-softmax-restorebest-smoke-dim4-context80/` |
| Representation option | `--use-prompt-prefix-projection` |
| Mode | `branch-target-softmax-unlikelihood` |
| Stabilizers | `--direct-answer-freeze-output-bias`, `--direct-answer-restore-best-branch-snapshot` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `50/50` |
| Prompt-prefix projection movement | all `20` parameters moved, max abs about `0.0942` |
| Composite train loss | `5.6649 -> 5.5679` |
| Restored best branch snapshot | yes, from step `40` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | all `"u"` |
| Final QA target-token coverage | `0.125` |
| Promotion status | rejected representation evidence |

Latest prompt-position projection smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.43-prompt-position-target-softmax-restorebest-smoke-dim4-context80/` |
| Representation option | `--use-prompt-position-projection` |
| Mode | `branch-target-softmax-unlikelihood` |
| Stabilizers | `--direct-answer-freeze-output-bias`, `--direct-answer-restore-best-branch-snapshot` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `50/50` |
| Prompt-position projection movement | `1108/1284` parameters moved, max abs about `0.0942` |
| Composite train loss | `5.6649 -> 5.5679` |
| Restored best branch snapshot | yes, from step `40` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | all `"u"` |
| Final QA target-token coverage | `0.125` |
| Promotion status | rejected representation evidence |

Latest branch-target margin smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.43-branch-target-margin-prompt-position-smoke-dim4-context80/` |
| Mode | `branch-target-margin-unlikelihood` |
| Representation option | `--use-prompt-position-projection` |
| Stabilizers | `--direct-answer-freeze-output-bias`, `--direct-answer-restore-best-branch-snapshot` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `50/50` |
| Prompt-position projection movement | `1108/1284` parameters moved, max abs about `0.1096` |
| Train loss | `4.8973 -> 4.7784` |
| Restored best branch snapshot | yes, from step `40` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | all `"u"` |
| Final QA target-token coverage | `0.125` |
| Promotion status | rejected target-margin evidence |

Latest branch-representation contrast smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.43-branch-representation-contrast50-prompt-position-smoke-dim4-context80/` |
| Mode | `branch-representation-contrast-unlikelihood` |
| Representation option | `--use-prompt-position-projection` |
| Representation contrast weight | `50.0` |
| Stabilizers | `--direct-answer-freeze-output-bias`, `--direct-answer-restore-best-branch-snapshot` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `50/50` |
| Snapshot diagnostic | `branch_representation_profiles` |
| Train loss | `53.5827 -> 53.4342` |
| Restored best branch snapshot | yes, from step `40` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | all `"u"` |
| Final QA target-token coverage | `0.125` |
| Final QA different-target hidden distance | avg about `0.00107`, max about `0.00237` |
| Promotion status | rejected representation-contrast evidence |

Latest branch-representation capacity smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.43-branch-representation-contrast50-prompt-position-smoke-dim8-context80-steps40/` |
| Mode | `branch-representation-contrast-unlikelihood` |
| Embedding/feed-forward dim | `8/16` |
| Representation option | `--use-prompt-position-projection` |
| Representation contrast weight | `50.0` |
| Stabilizers | `--direct-answer-freeze-output-bias`, `--direct-answer-restore-best-branch-snapshot` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `40/40`; 50-step dim8 screen was too slow for the regular loop |
| Train loss | `53.6111 -> 53.5752` |
| Restored best branch snapshot | yes, from step `10` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | all `"u"` |
| Final QA target-token coverage | `0.125` |
| Final QA different-target hidden distance | avg about `0.00209`, max about `0.00367` |
| Promotion status | rejected capacity evidence |

Latest prompt-position scale smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.43-prompt-position-scale32-repcontrast50-smoke-dim4-context80/` |
| Mode | `branch-representation-contrast-unlikelihood` |
| Representation option | `--use-prompt-position-projection` |
| Prompt-position scale | `32.0` |
| Representation contrast weight | `50.0` |
| Stabilizers | `--direct-answer-freeze-output-bias`, `--direct-answer-restore-best-branch-snapshot` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `50/50` |
| Train loss | `55.3835 -> 50.8435` |
| Prompt-position parameters moved | `1108/1284`, max absolute value about `0.07087` |
| Restored best branch snapshot | yes, from step `40` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | all `"u"` |
| Final QA target-token coverage | `0.125` |
| Final QA different-target hidden distance | restored avg about `0.01235`, max about `0.03610`; raw step-50 avg about `0.4115` before restore |
| Promotion status | rejected prompt-signal scale evidence |

Latest pre-layer-norm structural smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.44-prelayernorm-repcontrast50-prompt-position-smoke-dim4-context80/` |
| Mode | `branch-representation-contrast-unlikelihood` |
| Architecture option | `--use-pre-layer-norm` |
| Representation option | `--use-prompt-position-projection` |
| Representation contrast weight | `50.0` |
| Stabilizers | `--direct-answer-freeze-output-bias`, `--direct-answer-restore-best-branch-snapshot` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `50/50` |
| Train loss | final interval `43.8918` |
| Prompt-position parameters moved | `1108/1284`, max absolute value about `0.44679` |
| Final-norm parameters moved | `8/8`, max absolute value about `2.6389` |
| Restored best branch snapshot | no; step `50` was best |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | all `"y"` |
| Final QA target-token coverage | `0.125` |
| Final QA different-target hidden distance | avg about `0.2835`, max about `0.5151` |
| Partial diversity | `7/9` multi-target profiles no longer fully collapsed |
| Promotion status | partial structural evidence, rejected for promotion |

Latest target-balanced branch-batch smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.44-target-balanced-prelayernorm-repcontrast50-prompt-position-smoke-dim4-context80/` |
| Mode | `branch-balanced-representation-contrast-unlikelihood` |
| Architecture option | `--use-pre-layer-norm` |
| Representation option | `--use-prompt-position-projection` |
| Batch sampler | target-bucket balanced branch batch |
| Representation contrast weight | `50.0` |
| Stabilizers | `--direct-answer-freeze-output-bias`, `--direct-answer-restore-best-branch-snapshot` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `50/50` |
| Train loss | final interval `50.6619` |
| Prompt-position parameters moved | `516/1284`, max absolute value about `0.05881` |
| Final-norm parameters moved | `8/8`, max absolute value about `1.0013` |
| Restored best branch snapshot | yes, restored to baseline step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | all `"n"` |
| Final QA target-token coverage | `0.125` |
| Final QA different-target hidden distance | restored avg about `0.1261`, max about `0.2476` |
| Promotion status | rejected sampler evidence |

Latest branch-rank diagnostic smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.45-branch-rank-diagnostic-smoke-dim4-context80/` |
| Diagnostic | branch target rank, top-3/top-5 target coverage, and failed-record top predictions |
| Architecture option | `--use-pre-layer-norm` |
| Representation option | `--use-prompt-position-projection` |
| Snapshot mode | `branch-only` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `1/1` |
| Final QA dominant prediction | all `"n"` |
| Final QA average target rank | `14.25` |
| Final QA top-3/top-5 target coverage | `0.125` / `0.125` |
| Final heldout dominant prediction | all `"n"` |
| Final heldout average target rank | `14.25` |
| Final heldout top-3/top-5 target coverage | `0.125` / `0.125` |
| Promotion status | diagnostic evidence only |

Latest output-binding repair smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.46-output-binding-rankscore-smoke-dim4-context80/` |
| Mode | `branch-output-binding-unlikelihood` |
| Architecture option | `--use-pre-layer-norm` |
| Representation option | `--use-prompt-position-projection` |
| Binding weight | `2.0` |
| Stabilizers | `--direct-answer-freeze-output-bias`, rank-aware `--direct-answer-restore-best-branch-snapshot` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `20/20` |
| Train loss | `8.7064 -> 8.2205` |
| Restored best branch snapshot | no; step `20` was best by aggregate rank-aware score |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `2` |
| Final QA dominant predictions | wrong `"l"`/`"j"` branch tokens |
| Final QA average target rank | `17.375 -> 14.125` |
| Final QA top-3/top-5 target coverage | `0.125 -> 0.0` / `0.125 -> 0.25` |
| Final heldout average target rank | `17.25 -> 14.375` |
| Final heldout top-3/top-5 target coverage | `0.125 -> 0.0` / `0.125 -> 0.25` |
| Promotion status | rejected output-binding evidence |

Latest rank-margin repair smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.47-rank-margin-steps50-smoke-dim4-context80/` |
| Mode | `branch-rank-margin-unlikelihood` |
| Architecture option | `--use-pre-layer-norm` |
| Representation option | `--use-prompt-position-projection` |
| Hard wrong tokens | `5` |
| Margin weight | `2.0` |
| Stabilizers | `--direct-answer-freeze-output-bias`, rank-aware `--direct-answer-restore-best-branch-snapshot` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `50/50` |
| Train loss | `7.3649 -> 6.1629` |
| Restored best branch snapshot | yes, restored from step `40` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | wrong `"n"` |
| Final QA average target rank | `17.375 -> 9.0` |
| Final QA target-token coverage | `0.0 -> 0.125` |
| Final QA top-3/top-5 target coverage | `0.125 -> 0.25` / `0.125 -> 0.5` |
| Final heldout average target rank | `17.25 -> 9.0` |
| Final heldout target-token coverage | `0.0 -> 0.125` |
| Final heldout top-3/top-5 target coverage | `0.125 -> 0.25` / `0.125 -> 0.375` |
| Promotion status | rejected rank-lift evidence |

Latest balanced rank-margin repair smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.48-balanced-rank-margin-smoke-dim4-context80/` |
| Mode | `branch-balanced-rank-margin-unlikelihood` |
| Architecture option | `--use-pre-layer-norm` |
| Representation option | `--use-prompt-position-projection` |
| Batch sampler | target-balanced branch batch |
| Hard wrong tokens | `5` |
| Margin weight | `2.0` |
| Stabilizers | `--direct-answer-freeze-output-bias`, rank-aware `--direct-answer-restore-best-branch-snapshot` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `50/50` |
| Train loss | `7.2303 -> 6.3662` |
| Restored best branch snapshot | no; step `50` was best by aggregate rank-aware score |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `2` |
| Final QA dominant predictions | wrong `"a"`/`"n"` branch tokens |
| Final QA average target rank | `17.375 -> 9.375` |
| Final QA target-token coverage | `0.0 -> 0.125` |
| Final QA top-3/top-5 target coverage | `0.125 -> 0.375` / `0.125 -> 0.5` |
| Final heldout average target rank | `17.25 -> 9.625` |
| Final heldout target-token coverage | `0.0 -> 0.125` |
| Final heldout top-3/top-5 target coverage | `0.125 -> 0.25` / `0.125 -> 0.5` |
| Promotion status | rejected balanced rank-lift evidence |

Latest top-one hard-negative rank-margin smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.49-balanced-rank-margin-top1-smoke-dim4-context80/` |
| Mode | `branch-balanced-rank-margin-unlikelihood` |
| Architecture option | `--use-pre-layer-norm` |
| Representation option | `--use-prompt-position-projection` |
| Batch sampler | target-balanced branch batch |
| Hard wrong tokens | `1` |
| Margin weight | `2.0` |
| Stabilizers | `--direct-answer-freeze-output-bias`, rank-aware `--direct-answer-restore-best-branch-snapshot` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `50/50` |
| Train loss | `7.3512 -> 6.3642` |
| Restored best branch snapshot | yes, restored from step `10` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | wrong `"n"` |
| Final QA average target rank | `17.375 -> 12.5` |
| Final QA target-token coverage | `0.0 -> 0.125` |
| Final QA top-3/top-5 target coverage | `0.125 -> 0.125` / `0.125 -> 0.25` |
| Final heldout average target rank | `17.25 -> 12.375` |
| Final heldout target-token coverage | `0.0 -> 0.125` |
| Final heldout top-3/top-5 target coverage | `0.125 -> 0.125` / `0.125 -> 0.25` |
| Promotion status | rejected top-one hard-negative evidence |

Latest top-k softmax branch repair smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.50-balanced-topk-softmax-w5-smoke-dim4-context80/` |
| Mode | `branch-balanced-topk-softmax-unlikelihood` |
| Architecture option | `--use-pre-layer-norm` |
| Representation option | `--use-prompt-position-projection` |
| Batch sampler | target-balanced branch batch |
| Hard wrong tokens | `5` |
| Restricted-softmax weight | `5.0` |
| Stabilizers | `--direct-answer-freeze-output-bias`, rank-aware `--direct-answer-restore-best-branch-snapshot` |
| Context gate | passed, `219/219` semantic records covered |
| Direct steps | `50/50` |
| Restored best branch snapshot | yes, restored from step `40` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `1` |
| Final QA dominant prediction | wrong `"u"` |
| Final QA average target rank | `17.375 -> 8.75` |
| Final QA target-token coverage | `0.0 -> 0.125` |
| Final QA top-3/top-5 target coverage | `0.125 -> 0.375` / `0.125 -> 0.5` |
| Final heldout average target rank | `17.25 -> 8.75` |
| Final heldout target-token coverage | `0.0 -> 0.125` |
| Final heldout top-3/top-5 target coverage | `0.125 -> 0.375` / `0.125 -> 0.5` |
| Promotion status | rejected top-k softmax rank-lift evidence |

Latest foundation-stack smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-v0.51-foundation-stack-smoke/` |
| Checkpoint format | `quarklm-transformer-v2` |
| Optimizer | `adamw` with saved `optimizer_state.json` |
| Schedule / accumulation | warmup `1`, decay `2`, gradient accumulation `2` |
| Architecture switches | `--attention-heads 2`, `--use-rms-norm`, `--use-gated-mlp`, `--tie-output-embeddings`, `--use-rotary-positions` |
| Runtime switches | `--use-kv-cache-path`, eval `--use-kv-cache`, top-k/top-p/temperature/repetition controls |
| Eval artifacts | `eval.json` and replayable `eval_samples.jsonl` with token traces |
| Steps | `2/2` language-model smoke steps |
| Validation status | mechanics smoke completed; transformer tests pass |
| Promotion status | foundation mechanics evidence only |

Latest full-stack top-k branch repair smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.52-fullstack-topk-softmax-smoke-dim4-context80/` |
| Mode | `branch-balanced-topk-softmax-unlikelihood` |
| Foundation stack | AdamW, gradient accumulation, two heads, RMSNorm, gated MLP, tied output embeddings, rotary positions, cache-aware metadata |
| Context / representation | context `80`, `--use-pre-layer-norm`, `--use-prompt-position-projection` |
| Direct steps | `50/50` |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` |
| Final QA dominant prediction | wrong `"i"` |
| Final QA average target rank | `13.25` |
| Final QA target-token coverage | `0.25` |
| Final QA top-3/top-5 target coverage | `0.25` / `0.375` |
| Final heldout average target rank | `13.375` |
| Final heldout target-token coverage | `0.25` |
| Final heldout top-3/top-5 target coverage | `0.25` / `0.375` |
| Promotion status | rejected unchanged top-k pressure under full stack |

Latest full-stack bidirectional binding branch repair smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.53-fullstack-bidir-binding-smoke-dim4-context80/` |
| Mode | `branch-balanced-bidirectional-binding-unlikelihood` |
| Foundation stack | AdamW, gradient accumulation, two heads, RMSNorm, gated MLP, tied output embeddings, rotary positions, cache-aware metadata |
| Context / representation | context `80`, `--use-pre-layer-norm`, `--use-prompt-position-projection` |
| Binding pressure | row-wise branch target choice plus column-wise target-token ownership across prompt contexts |
| Unit coverage | focused transformer tests pass, including the context-ownership regression |
| Direct steps | `50/50` |
| Restored best branch snapshot | yes, restored from step `40` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `2` |
| Final QA dominant prediction | wrong `"a"` |
| Final QA average target rank | `7.875` |
| Final QA target-token coverage | `0.125` |
| Final QA top-3/top-5 target coverage | `0.25` / `0.5` |
| Step-50 QA note | target-token coverage briefly reached `0.25` with average rank `8.375` before restore selected step `40` |
| Final heldout average target rank | `9.0` |
| Final heldout target-token coverage | `0.125` |
| Final heldout top-3/top-5 target coverage | `0.25` / `0.375` |
| Promotion status | partial rank-pressure progress; rejected until target coverage is preserved and top-1 branch choices improve |

Latest full-stack coverage binding branch repair smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.54-fullstack-coverage-binding-smoke-dim4-context80/` |
| Mode | `branch-balanced-coverage-binding-unlikelihood` |
| Foundation stack | AdamW, gradient accumulation, two heads, RMSNorm, gated MLP, tied output embeddings, rotary positions, cache-aware metadata |
| Context / representation | context `80`, `--use-pre-layer-norm`, `--use-prompt-position-projection` |
| Binding pressure | branch targets versus sibling targets plus hard wrong tokens, with target-set mass coverage guard |
| Hard wrong tokens | `5` |
| Unit coverage | focused transformer tests pass, including the hard-wrong-token coverage regression |
| Direct steps | `50/50` |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` |
| Final QA dominant prediction | wrong `"i"` |
| Final QA average target rank | `13.25` |
| Final QA target-token coverage | `0.25` |
| Final QA top-3/top-5 target coverage | `0.25` / `0.375` |
| Training snapshot note | step `50` improved QA average target rank to `8.125`, but target-token coverage collapsed to `0.0` with wrong `"a"` top-1 collapse |
| Final heldout average target rank | `13.375` |
| Final heldout target-token coverage | `0.25` |
| Final heldout top-3/top-5 target coverage | `0.25` / `0.375` |
| Promotion status | rejected; best-snapshot scoring protected the checkpoint, but the objective traded target coverage away for rank |

Latest full-stack target-set coverage branch repair smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.55-fullstack-target-set-coverage-smoke-dim4-context80/` |
| Mode | `branch-balanced-target-set-coverage-unlikelihood` |
| Foundation stack | AdamW, gradient accumulation, two heads, RMSNorm, gated MLP, tied output embeddings, rotary positions, cache-aware metadata |
| Context / representation | context `80`, `--use-pre-layer-norm`, `--use-prompt-position-projection` |
| Coverage pressure | branch target set versus hard wrong tokens, without exact-target row loss or cross-context ownership |
| Positive target CE | `0.0` |
| Hard wrong tokens | `5` |
| Unit coverage | focused transformer tests pass, including the target-set-only coverage regression |
| Direct steps | `50/50` |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` |
| Final QA dominant prediction | wrong `"i"` |
| Final QA average target rank | `13.25` |
| Final QA target-token coverage | `0.25` |
| Final QA top-3/top-5 target coverage | `0.25` / `0.375` |
| Training snapshot note | step `50` improved QA average target rank to `10.0`, but target-token coverage collapsed to `0.0` with wrong `"a"` top-1 collapse |
| Final heldout average target rank | `13.375` |
| Final heldout target-token coverage | `0.25` |
| Final heldout top-3/top-5 target coverage | `0.25` / `0.375` |
| Promotion status | rejected; batch-local target-set mass is not enough to preserve eval target-token coverage |

Latest full-stack target-diversity branch repair smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.57-fullstack-target-diversity-smoke-dim4-context80/` |
| Mode | `branch-balanced-target-diversity-unlikelihood` |
| Foundation stack | AdamW, gradient accumulation, two heads, RMSNorm, gated MLP, tied output embeddings, rotary positions, cache-aware metadata |
| Context / representation | context `80`, `--use-pre-layer-norm`, `--use-prompt-position-projection` |
| Diversity pressure | target-set mass plus target-share balance over branch targets |
| Positive target CE | `0.0` |
| Hard wrong tokens | `5` |
| Unit coverage | focused transformer tests pass, including restricted target-set mass and weakest target-share balance regression |
| Direct steps | `50/50` |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` |
| Final QA dominant prediction | wrong `"i"` |
| Final QA average target rank | `13.25` |
| Final QA target-token coverage | `0.25` |
| Final QA top-3/top-5 target coverage | `0.25` / `0.375` |
| Training snapshot note | step `50` improved QA average target rank to `10.0`, but target-token coverage collapsed to `0.0` with wrong `"a"` top-1 collapse |
| Final heldout average target rank | `13.375` |
| Final heldout target-token coverage | `0.25` |
| Final heldout top-3/top-5 target coverage | `0.25` / `0.375` |
| Promotion status | rejected; batch-local target-share diversity still does not preserve eval-wide target-token coverage |

Latest full-stack target-replay coverage branch repair smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.58-fullstack-target-replay-coverage-smoke-dim4-context80/` |
| Mode | `branch-balanced-target-replay-coverage-unlikelihood` |
| Foundation stack | AdamW, gradient accumulation, two heads, RMSNorm, gated MLP, tied output embeddings, rotary positions, cache-aware metadata |
| Context / representation | context `80`, `--use-pre-layer-norm`, `--use-prompt-position-projection` |
| Replay pressure | target-set mass plus target-share balance over admitted branch-pool targets |
| Positive target CE | `0.0` |
| Hard wrong tokens | `5` |
| Unit coverage | focused transformer tests pass, including sampled-batch missing pool-target replay regression |
| Direct steps | `50/50` |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` |
| Final QA dominant prediction | wrong `"i"` |
| Final QA average target rank | `13.25` |
| Final QA target-token coverage | `0.25` |
| Final QA top-3/top-5 target coverage | `0.25` / `0.375` |
| Training snapshot note | step `40` improved QA average target rank to `6.875` and top-5 coverage to `0.5`; by step `50`, QA/heldout top-1 collapsed to wrong `"n"` and target-token coverage had hit `0.0` during training |
| Final heldout average target rank | `13.375` |
| Final heldout target-token coverage | `0.25` |
| Final heldout top-3/top-5 target coverage | `0.25` / `0.375` |
| Promotion status | rejected; pool-owned replay coverage still does not preserve context-specific target ownership |

Latest full-stack context-replay coverage branch repair smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.59-fullstack-context-replay-coverage-smoke-dim4-context80/` |
| Mode | `branch-balanced-context-replay-coverage-unlikelihood` |
| Foundation stack | AdamW, gradient accumulation, two heads, RMSNorm, gated MLP, tied output embeddings, rotary positions, cache-aware metadata |
| Context / representation | context `80`, `--use-pre-layer-norm`, `--use-prompt-position-projection` |
| Replay pressure | target-set mass plus context-owned target share over admitted branch-pool replay contexts |
| Positive target CE | `0.0` |
| Hard wrong tokens | `5` |
| Unit coverage | focused transformer tests pass, including fixed replay-context owned-target share regression |
| Direct steps | `50/50` |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` |
| Final QA dominant prediction | wrong `"i"` |
| Final QA average target rank | `13.25` |
| Final QA target-token coverage | `0.25` |
| Final QA top-3/top-5 target coverage | `0.25` / `0.375` |
| Training snapshot note | step `40` improved QA average target rank to `7.375`, top-3 to `0.375`, and top-5 to `0.5`; by step `50`, QA predicted diversity was only `2/8` and target-token coverage had hit `0.0` during training |
| Final heldout average target rank | `13.375` |
| Final heldout target-token coverage | `0.25` |
| Final heldout top-3/top-5 target coverage | `0.25` / `0.375` |
| Promotion status | rejected; context-owned replay improves rank/top-k snapshots but still does not preserve target-token coverage |

Latest full-stack coverage-floor branch restore smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.60-fullstack-context-replay-coverage-floor-metadata-smoke-dim4-context80/` |
| Mode | `branch-balanced-context-replay-coverage-unlikelihood` |
| Scoring guard | profile-wise target-token coverage floor before rank/top-k scoring |
| Snapshot metadata | direct-answer JSONL rows include `branch_target_coverage_by_profile` |
| Foundation stack | AdamW, gradient accumulation, two heads, RMSNorm, gated MLP, tied output embeddings, rotary positions, cache-aware metadata |
| Context / representation | context `80`, `--use-pre-layer-norm`, `--use-prompt-position-projection` |
| Positive target CE | `0.0` |
| Hard wrong tokens | `5` |
| Unit coverage | focused transformer tests pass, including profile-wise coverage-floor regression |
| Direct steps | `50/50` |
| Direct-answer JSONL rows | `7` clean rows |
| Restored best branch snapshot | yes, restored from step `0` |
| Baseline coverage floor | `qa` `0.25`, `heldout` `0.25`, `admissions` `0.1429`, minimum profile `0.0714` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` |
| Final QA dominant prediction | wrong `"i"` |
| Final QA average target rank | `13.25` |
| Final QA target-token coverage | `0.25` |
| Final QA top-3/top-5 target coverage | `0.25` / `0.375` |
| Training snapshot note | step `40` improved QA average target rank to `7.375`, top-3 to `0.375`, and top-5 to `0.5`, but regressed profile coverage and was ineligible for restore |
| Final heldout average target rank | `13.375` |
| Final heldout target-token coverage | `0.25` |
| Final heldout top-3/top-5 target coverage | `0.25` / `0.375` |
| Promotion status | gate repair accepted; trained model behavior rejected because coverage still collapses during training |

Latest full-stack covered-target anchor branch repair smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.61-fullstack-context-coverage-anchor-smoke-dim4-context80/` |
| Mode | `branch-balanced-context-coverage-anchor-unlikelihood` |
| Scoring guard | profile-wise target-token coverage floor before rank/top-k scoring |
| Anchor pressure | covered replay branches add target-vs-replay-target/hard-wrong CE |
| Foundation stack | AdamW, gradient accumulation, two heads, RMSNorm, gated MLP, tied output embeddings, rotary positions, cache-aware metadata |
| Context / representation | context `80`, `--use-pre-layer-norm`, `--use-prompt-position-projection` |
| Positive target CE | `0.0` |
| Hard wrong tokens | `5` |
| Unit coverage | focused transformer tests pass, including anchored-vs-unanchored covered branch regression |
| Direct steps | `50/50` |
| Direct-answer JSONL rows | `7` clean rows |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` |
| Final QA dominant prediction | wrong `"i"` |
| Final QA average target rank | `13.25` |
| Final QA target-token coverage | `0.25` |
| Final QA top-3/top-5 target coverage | `0.25` / `0.375` |
| Training snapshot note | snapshots collapsed harder to covered wrong `"i"`; QA/heldout predicted diversity fell to `1/8`, target-token coverage to `0.125`, and average target rank above `21` |
| Final heldout average target rank | `13.375` |
| Final heldout target-token coverage | `0.25` |
| Final heldout top-3/top-5 target coverage | `0.25` / `0.375` |
| Promotion status | rejected; global covered-target anchoring over-protects one covered token instead of preserving coverage diversity |

Latest full-stack coverage-preserving deficit branch repair smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.65-fullstack-coverage-preserving-deficit-smoke-dim4-context80/` |
| Mode | `branch-balanced-context-coverage-preserving-deficit-unlikelihood` |
| Scoring guard | profile-wise target-token coverage floor before rank/top-k scoring |
| Deficit pressure | replay target tokens absent from current replay predictions receive target-vs-hard-candidate pressure |
| Preservation pressure | target tokens currently represented in replay predictions receive target-balanced anchors |
| Foundation stack | AdamW, gradient accumulation, two heads, RMSNorm, gated MLP, tied output embeddings, rotary positions, cache-aware metadata |
| Context / representation | context `80`, `--use-pre-layer-norm`, `--use-prompt-position-projection` |
| Positive target CE | `0.0` |
| Hard wrong tokens | `5` |
| Unit coverage | focused transformer tests pass, including missing-target lift and represented-target preservation regressions |
| Direct steps | `50/50` |
| Direct-answer JSONL rows | `7` clean rows |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` |
| Final QA dominant prediction | wrong `"i"` |
| Final QA average target rank | `13.25` |
| Final QA target-token coverage | `0.25` |
| Final QA top-3/top-5 target coverage | `0.25` / `0.375` |
| Training snapshot note | step `50` reached QA/heldout branch accuracy `1/8`, QA average target rank `7.75`, heldout average target rank `7.125`, and top-5 coverage `0.5`, but both profiles collapsed to predicted diversity `1/8` and target-token coverage `0.125` |
| Final heldout average target rank | `13.375` |
| Final heldout target-token coverage | `0.25` |
| Final heldout top-3/top-5 target coverage | `0.25` / `0.375` |
| Promotion status | rejected; current-prediction preservation improves rank but over-preserves one represented target token |

Latest profile-aware replay-plan smoke:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.67-profile-aware-replay-plan-smoke-dim4-context80/` |
| Mode | `branch-balanced-context-profile-coverage-preserving-deficit-unlikelihood` |
| Replay plan artifact | `direct_answer_replay_plan.json` |
| Replay plan size | `9144` branch records and `9144` replay records across `21` profiles |
| Example profile floors | `qa:place` coverage floor `0.5`; `qa:color` coverage floor `0.0` |
| Foundation stack | AdamW, two heads, RMSNorm, gated MLP, tied output embeddings, rotary positions, cache-aware metadata |
| Context / representation | context `80`, `--use-pre-layer-norm`, `--use-prompt-position-projection` |
| Unit coverage | focused transformer tests pass, including profile-deficit isolation and shared-target source preservation |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Direct steps | `1/1` bounded smoke |
| Snapshot mode | `branch-only`; post-direct candidate snapshot skipped and recorded |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Promotion status | mechanics-readiness evidence only; profile-aware plan exists, but model quality is not promoted |

Latest profile-aware full-stack repair screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.68-fullstack-profile-aware-preserving-deficit-smoke-dim4-context80/` |
| Mode | `branch-balanced-context-profile-coverage-preserving-deficit-unlikelihood` |
| Replay plan artifact | `direct_answer_replay_plan.json` |
| Replay plan size | `9144` branch records and `9144` replay records across `21` profiles |
| Foundation stack | AdamW, two heads, RMSNorm, gated MLP, tied output embeddings, rotary positions, cache-aware metadata |
| Context / representation | context `80`, `--use-pre-layer-norm`, `--use-prompt-position-projection` |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Direct steps | `50/50` |
| Direct-answer JSONL rows | `7` clean rows |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` after restore |
| Final QA average target rank | `13.25` after restore |
| Final QA target-token coverage | `0.25` after restore |
| Training snapshot note | step `40` improved QA average target rank to `6.5` and top-5 coverage to `0.625`, but QA target-token coverage regressed to `0.125` and predicted diversity collapsed to `1/8` |
| Final heldout average target rank | `13.375` after restore |
| Training heldout note | step `40` improved heldout average target rank to `6.875` and top-5 coverage to `0.5`, but target-token coverage regressed to `0.125` and predicted diversity collapsed to `1/8` |
| Promotion status | rejected; profile-aware rank gains still trade away coverage and diversity |

Latest profile target-share full-stack screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.82-fullstack-profile-target-share-smoke-dim4-context80/` |
| Mode | `branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood` |
| Artifact stack | experiment intent, corpus hygiene, training plan, candidate quarantine, deterministic verifier, recipe, replay plan, constraint-first report, metrics, tokenizer, optimizer, lessons, checkpoint |
| Replay plan size | `9144` branch records and `9144` replay records across `21` profiles |
| Foundation stack | AdamW, two heads, RMSNorm, gated MLP, tied output embeddings, rotary positions, cache-aware metadata |
| Context / representation | context `80`, `--use-pre-layer-norm`, `--use-prompt-position-projection` |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Purity gates | no pretrained weights, no pretrained tokenizer, no external embeddings |
| Direct steps | `50/50` |
| Direct-answer JSONL rows | `7` clean rows |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` after restore |
| Final QA average target rank | `13.25` after restore |
| Final QA target-token coverage | `0.25` after restore |
| Training snapshot note | step `40` improved QA average target rank to `9.125` and top-5 coverage to `0.375`, but QA target-token coverage regressed to `0.0` and predicted diversity collapsed to `1/8` |
| Training heldout note | step `40` improved heldout average target rank to `9.25` and top-5 coverage to `0.375`, but heldout target-token coverage regressed to `0.0` and predicted diversity collapsed to `1/8` |
| Promotion status | rejected; target-share pressure still trades coverage and diversity away for rank |

Latest prompt-specific branch ownership full-stack screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.83-fullstack-prompt-ownership-smoke-dim4-context80/` |
| Mode | `branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood` |
| Added mechanic | sibling-target margin inside each profile so a replay context is trained to outrank other profile targets |
| Unit coverage | focused transformer test passes; prompt ownership lifts a context-specific target more than v0.82 target-share pressure |
| Artifact stack | experiment intent, corpus hygiene, training plan, candidate quarantine, deterministic verifier, recipe, replay plan, constraint-first report, metrics, tokenizer, optimizer, lessons, checkpoint |
| Replay plan size | `9144` branch records and `9144` replay records across `21` profiles |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Purity gates | no pretrained weights, no pretrained tokenizer, no external embeddings |
| Direct steps | `50/50` |
| Direct-answer JSONL rows | `7` clean rows |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` after restore |
| Final QA average target rank | `13.25` after restore |
| Final QA target-token coverage | `0.25` after restore |
| Training snapshot note | step `50` improved QA average target rank to `8.625`, but QA target-token coverage regressed to `0.0` and predicted diversity collapsed to `1/8` |
| Training heldout note | step `50` improved heldout average target rank to `8.5`, but heldout target-token coverage regressed to `0.0` and predicted diversity collapsed to `1/8` |
| Promotion status | rejected; prompt ownership needs coverage-preserving training before rank gains can be trusted |

Latest baseline-anchored prompt ownership full-stack screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.84-fullstack-baseline-anchored-prompt-ownership-smoke-dim4-context80/` |
| Mode | `branch-balanced-context-profile-baseline-anchored-prompt-ownership-target-share-preserving-deficit-unlikelihood` |
| Added mechanic | replay preservation uses baseline profile-aware replay predictions instead of current prediction drift |
| Unit coverage | focused transformer tests pass; baseline prediction overrides are used by profiled replay batches and protect a covered target better than dynamic prediction preservation |
| Artifact stack | experiment intent, corpus hygiene, training plan, candidate quarantine, deterministic verifier, recipe, replay plan, constraint-first report, metrics, tokenizer, optimizer, lessons, checkpoint |
| Replay plan size | `9144` branch records and `9144` replay records across `21` profiles |
| Baseline prediction anchors | `562` recorded and active |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Purity gates | no pretrained weights, no pretrained tokenizer, no external embeddings |
| Direct steps | `50/50` |
| Direct-answer JSONL rows | `7` clean rows |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` after restore |
| Final QA average target rank | `13.25` after restore |
| Final QA target-token coverage | `0.25` after restore |
| Training snapshot note | step `40` improved QA average target rank to `8.0`, but QA target-token coverage regressed to `0.125` and predicted diversity collapsed to `1/8` |
| Training heldout note | step `40` improved heldout average target rank to `8.375`, but heldout target-token coverage regressed to `0.125` and predicted diversity collapsed to `1/8` |
| Promotion status | rejected; baseline anchors improve coverage over v0.83 but still miss the full `0.25` coverage floor |

Latest baseline-floor update-gated prompt ownership full-stack screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.85-fullstack-baseline-floor-gated-prompt-ownership-smoke-dim4-context80/` |
| Mode | `branch-balanced-context-profile-baseline-floor-gated-prompt-ownership-target-share-preserving-deficit-unlikelihood` |
| Added mechanic | direct-answer updates are rolled back when branch-profile target-token coverage falls below the step-0 baseline floor |
| Unit coverage | focused transformer tests pass; the new mode records active baseline replay anchors and update-guard accounting |
| Artifact stack | experiment intent, corpus hygiene, training plan, candidate quarantine, deterministic verifier, recipe, replay plan, constraint-first report, metrics, tokenizer, optimizer, lessons, checkpoint |
| Replay plan size | `9144` branch records and `9144` replay records across `21` profiles |
| Baseline prediction anchors | `562` recorded and active |
| Update guard | checked `50/50` attempted updates; accepted `0`; rejected `50` |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Purity gates | no pretrained weights, no pretrained tokenizer, no external embeddings |
| Direct steps | `50/50` attempted |
| Direct-answer JSONL rows | `7` clean rows |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` |
| Final QA average target rank | `13.25` |
| Final QA target-token coverage | `0.25` |
| Training snapshot note | every recorded trained snapshot preserved QA target-token coverage at `0.25`, but only because every attempted update was rejected |
| Training heldout note | every recorded trained snapshot preserved heldout target-token coverage at `0.25`, but only because every attempted update was rejected |
| Promotion status | rejected; the guard prevents unsafe forgetting, but no weight update is accepted and branch diversity still fails |

Latest adaptive baseline-floor prompt ownership full-stack screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.86-fullstack-baseline-floor-adaptive-prompt-ownership-smoke-dim4-context80/` |
| Mode | `branch-balanced-context-profile-baseline-floor-adaptive-prompt-ownership-target-share-preserving-deficit-unlikelihood` |
| Added mechanic | rejected updates are retried at learning-rate scales `1.0`, `0.25`, `0.05`, and `0.01` after restoring model, optimizer, and RNG state |
| Unit coverage | focused transformer tests pass; the new mode records active baseline replay anchors and adaptive retry accounting |
| Artifact stack | experiment intent, corpus hygiene, training plan, candidate quarantine, deterministic verifier, recipe, replay plan, constraint-first report, metrics, tokenizer, optimizer, lessons, checkpoint |
| Replay plan size | `9144` branch records and `9144` replay records across `21` profiles |
| Baseline prediction anchors | `562` recorded and active |
| Adaptive scales | `1.0`, `0.25`, `0.05`, `0.01` |
| Update guard | checked `50/50` steps; attempted `200` scaled updates; accepted `0`; rejected `200` |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Purity gates | no pretrained weights, no pretrained tokenizer, no external embeddings |
| Direct steps | `50/50` attempted |
| Direct-answer JSONL rows | `7` clean rows |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` |
| Final QA average target rank | `13.25` |
| Final QA target-token coverage | `0.25` |
| Training snapshot note | every recorded trained snapshot preserved QA target-token coverage at `0.25`, but adaptive retries accepted no updates |
| Training heldout note | every recorded trained snapshot preserved heldout target-token coverage at `0.25`, but adaptive retries accepted no updates |
| Promotion status | rejected; smaller learning-rate scales do not make the update safe, which sets up the v0.87 repair-retry screen |

Latest repaired baseline-floor prompt ownership full-stack screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.87-fullstack-baseline-floor-repaired-prompt-ownership-clean-smoke-dim4-context80/` |
| Mode | `branch-balanced-context-profile-baseline-floor-repaired-prompt-ownership-target-share-preserving-deficit-unlikelihood` |
| Added mechanic | failed adaptive retries get one bounded baseline-covered anchor repair before the floor probe decides whether to keep or roll back the update |
| Unit coverage | focused transformer tests pass; the new mode records active baseline replay anchors, repair anchors, repair attempts, and accepted update-shape accounting |
| Artifact stack | experiment intent, corpus hygiene, training plan, candidate quarantine, deterministic verifier, recipe, replay plan, constraint-first report, metrics, tokenizer, optimizer, lessons, checkpoint |
| Replay plan size | `9144` branch records and `9144` replay records across `21` profiles |
| Baseline prediction anchors | `562` recorded and active |
| Repair anchors | `227` recorded; one repair step per failed retry |
| Adaptive scales | `1.0`, `0.25`, `0.05`, `0.01` |
| Update guard | checked `50/50` steps; attempted `200` updates; ran `200` one-step repairs; accepted `0`; rejected `200` |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Purity gates | no pretrained weights, no pretrained tokenizer, no external embeddings |
| Direct steps | `50/50` attempted |
| Direct-answer JSONL rows | `7` clean rows |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` |
| Final QA average target rank | `13.25` |
| Final QA target-token coverage | `0.25` |
| Training snapshot note | every recorded trained snapshot preserved QA target-token coverage at `0.25`, but repair retries accepted no updates |
| Training heldout note | every recorded trained snapshot preserved heldout target-token coverage at `0.25`, but repair retries accepted no updates |
| Promotion status | rejected; post-update repair is insufficient and the next repair needs a floor-preserving objective before optimizer application |

Latest objective-side baseline-floor prompt ownership full-stack screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.88-fullstack-baseline-floor-objective-prompt-ownership-smoke-dim4-context80/` |
| Mode | `branch-balanced-context-profile-baseline-floor-objective-prompt-ownership-target-share-preserving-deficit-unlikelihood` |
| Added mechanic | a balanced batch of baseline-covered floor anchors is included in the same direct-answer loss and backward pass as branch-diversity pressure |
| Unit coverage | focused transformer tests pass; the new mode records objective anchor counts, anchor batch size, anchor weight, and accepted/rejected guard accounting |
| Artifact stack | experiment intent, corpus hygiene, training plan, candidate quarantine, deterministic verifier, recipe, replay plan, constraint-first report, metrics, tokenizer, optimizer, lessons, checkpoint |
| Replay plan size | `9144` branch records and `9144` replay records across `21` profiles |
| Baseline prediction anchors | `562` recorded and active |
| Objective-side floor anchors | `227` recorded; batch size `32`; weight `10.0` |
| Adaptive scales | `1.0`, `0.25`, `0.05`, `0.01` |
| Update guard | checked `50/50` steps; attempted `200` updates; ran `200` objective anchor batches covering `2400` anchor records; accepted `0`; rejected `200` |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Purity gates | no pretrained weights, no pretrained tokenizer, no external embeddings |
| Direct steps | `50/50` attempted |
| Direct-answer JSONL rows | `7` clean rows |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` |
| Final QA average target rank | `13.25` |
| Final QA target-token coverage | `0.25` |
| Training snapshot note | every recorded trained snapshot preserved QA target-token coverage at `0.25`, but objective-side floor anchors accepted no updates |
| Training heldout note | every recorded trained snapshot preserved heldout target-token coverage at `0.25`, but objective-side floor anchors accepted no updates |
| Promotion status | rejected; the combined floor-anchor and branch-pressure objective is insufficient, which sets up the stabilization-only screen |

Latest stabilization-only baseline-floor full-stack screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.89-fullstack-baseline-floor-stabilization-smoke-dim4-context80/` |
| Mode | `branch-context-profile-baseline-floor-stabilization-unlikelihood` |
| Added mechanic | guarded attempts train only baseline-covered floor anchors, with branch-diversity pressure removed from the update shape |
| Unit coverage | focused transformer tests pass; the new mode records stabilization anchor counts, anchor batch size, stabilization batches, and accepted/rejected guard accounting |
| Artifact stack | experiment intent, corpus hygiene, training plan, candidate quarantine, deterministic verifier, recipe, replay plan, constraint-first report, metrics, tokenizer, optimizer, lessons, checkpoint |
| Replay plan size | `9144` branch records and `9144` replay records across `21` profiles |
| Baseline prediction anchors | `562` recorded and active |
| Stabilization floor anchors | `227` recorded; batch size `32` |
| Adaptive scales | `1.0`, `0.25`, `0.05`, `0.01` |
| Update guard | checked `50/50` steps; attempted `200` updates; ran `200` stabilization anchor batches covering `2400` anchor records; accepted `0`; rejected `200` |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Purity gates | no pretrained weights, no pretrained tokenizer, no external embeddings |
| Direct steps | `50/50` attempted |
| Direct-answer JSONL rows | `7` clean rows |
| Restored best branch snapshot | yes, restored from step `0` |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Final QA target/predicted unique | `8` / `3` |
| Final QA average target rank | `13.25` |
| Final QA target-token coverage | `0.25` |
| Training snapshot note | every recorded trained snapshot preserved QA target-token coverage at `0.25`, but stabilization-only floor anchors accepted no updates |
| Training heldout note | every recorded trained snapshot preserved heldout target-token coverage at `0.25`, but stabilization-only floor anchors accepted no updates |
| Promotion status | rejected; floor-only anchor updates are insufficient under the current guard, so the next repair should diagnose the guard/update interaction before branch pressure is added back |

Latest baseline-floor rejection diagnostics screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.90-fullstack-baseline-floor-stabilization-diagnostics-smoke-dim4-context80/` |
| Mode | `branch-context-profile-baseline-floor-stabilization-unlikelihood` |
| Added mechanic | guard records rejected update-shape counts, rejected scale counts, violation profile counts, diagnostic samples, and worst rejected floor violation |
| Unit coverage | focused transformer tests pass; the reusable coverage diagnostic helper reports profile deficits and the stabilization guard records rejection diagnostics |
| Update guard | checked `50/50` steps; attempted `200` updates; accepted `0`; rejected `200` |
| Rejected update shapes | `stabilization: 200` |
| Rejected adaptive scales | `1: 50`, `0.25: 50`, `0.05: 50`, `0.01: 50` |
| Violation profile counts | `heldout: 200`, `admissions: 150`, `glossary: 150`, `qa: 150`, `self: 100`, `learning: 50`, `owner: 50` |
| Worst rejected floor violation | `learning`, baseline coverage `0.25`, snapshot coverage `0.0`, deficit `0.25` |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Deterministic verifier | passed with no external model |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Promotion status | rejected for model promotion, but diagnostic evidence is usable for the next profile-targeted floor repair |

Profile-targeted baseline-floor stabilization screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.91-fullstack-baseline-floor-profile-targeted-stabilization-smoke-dim4-context80/` |
| Mode | `branch-context-profile-baseline-floor-profile-targeted-stabilization-unlikelihood` |
| Added mechanic | guarded attempts train the full baseline-covered floor-anchor profile-target surface instead of a random 32-anchor sample |
| Unit coverage | focused transformer tests pass; the new mode records profile-target activity, full floor batch sizing, profile-target counts, and source-profile anchor counts |
| Floor anchors | `227` recorded; requested batch size `227`; `12` profile-target groups |
| Anchor profile counts | `qa:owner 48`, `qa:place 41`, `fact:owner 40`, `fact:place 40`, `bridge:owner 20`, `bridge:place 16`, `fact:learning 8`, `qa:glossary 6`, `qa:learning 5`, `qa:self 3` |
| Update guard | checked `50/50` steps; attempted `200` updates; accepted `0`; rejected `200` |
| Rejected update shapes | `profile_targeted_stabilization: 200` |
| Rejected adaptive scales | `1: 50`, `0.25: 50`, `0.05: 50`, `0.01: 50` |
| Violation profile counts | `heldout: 200`, `admissions: 150`, `glossary: 150`, `qa: 150`, `self: 100`, `learning: 50`, `owner: 50` |
| Worst rejected floor violation | `learning`, baseline coverage `0.25`, snapshot coverage `0.0`, deficit `0.25` |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Deterministic verifier | passed with no external model |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Promotion status | rejected; full profile-target floor coverage alone does not make guarded updates safe |

Sequential profile-floor stabilization screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.92-fullstack-baseline-floor-sequential-profile-stabilization-smoke-dim4-context80/` |
| Mode | `branch-context-profile-baseline-floor-sequential-profile-stabilization-unlikelihood` |
| Added mechanic | guarded attempts train source-profile floor-anchor groups sequentially and roll back each unsafe group before trying the next one |
| Unit coverage | focused transformer tests pass; the new mode records sequential profile attempts, accept/reject counts, no-effective-update attempts, and profile probe samples |
| Floor anchors | `227` recorded; requested batch size `227`; `12` profile-target groups; `10` source-profile groups |
| Sequential profile attempts | `2000` attempted; `0` accepted; `2000` rejected; `2400` anchor records |
| Source-profile rejection counts | each of `bridge:owner`, `bridge:place`, `fact:learning`, `fact:owner`, `fact:place`, `qa:glossary`, `qa:learning`, `qa:owner`, `qa:place`, and `qa:self` rejected `200` times |
| Update guard | checked `50/50` steps; attempted `200` updates; accepted `0`; rejected `200`; no-effective-update attempts `200` |
| Rejected update shapes | `sequential_profile_stabilization: 200` |
| Rejected adaptive scales | `1: 50`, `0.25: 50`, `0.05: 50`, `0.01: 50` |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Deterministic verifier | passed with no external model |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Promotion status | rejected; sequential source-profile repair still cannot produce safe weight movement |

Calibrated sequential profile-floor stabilization screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.93-baseline-floor-calibrated-sequential-profile-stabilization-step1-dim4-context80/` |
| Mode | `branch-context-profile-baseline-floor-calibrated-sequential-profile-stabilization-unlikelihood` |
| Added mechanic | calibrated adaptive scales below `0.01` plus coverage-only guard probes for floor checks |
| Unit coverage | focused transformer tests pass; the mode records calibrated activation, extended scale metadata, replay-plan scales, and accepted/rejected update-shape accounting |
| Calibrated scales | `1`, `0.25`, `0.05`, `0.01`, `0.0025`, `0.0005`, `0.0001` |
| Update guard | checked `1/1` step; attempted `5` updates; accepted `1`; rejected `4`; no-effective-update attempts `4` |
| Accepted update | `bridge:owner` source-profile group at scale `0.0025` |
| Sequential profile attempts | `50` attempted; `1` accepted; `49` rejected; `60` anchor records |
| Rejected adaptive scales | `1: 1`, `0.25: 1`, `0.05: 1`, `0.01: 1` |
| Accepted update shapes | `calibrated_sequential_profile_stabilization: 1` |
| Rejected update shapes | `calibrated_sequential_profile_stabilization: 4` |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Deterministic verifier | passed with no external model |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Promotion status | rejected for model promotion; calibrated floor-preserving movement is now proven possible |

Latest profile-scale calibrated floor stabilization screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.94-baseline-floor-profile-scale-calibrated-sequential-stabilization-step1-dim4-context80/` |
| Mode | `branch-context-profile-baseline-floor-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood` |
| Added mechanic | profile-scale memory: search calibrated scales per source profile, preserve the first safe profile update, and roll back unsafe profile-scale attempts |
| Unit coverage | focused transformer tests pass; the mode records profile-scale activation, search/outer scales, profile-scale attempts, acceptance/rejection scale counts, and accepted profile scales |
| Search scales | `1`, `0.25`, `0.05`, `0.01`, `0.0025`, `0.0005`, `0.0001` |
| Outer guard | checked `1/1` step; attempted `1` update; accepted `1`; rejected `0`; no-effective-update attempts `0` |
| Profile-scale attempts | `60` attempted; `8` accepted; `52` rejected; `72` anchor records |
| Accepted profile scales | `bridge:owner 0.0025`, `bridge:place 0.0005`, `fact:learning 0.0005`, `fact:owner 0.0001`, `fact:place 0.0001`, `qa:glossary 0.0001`, `qa:place 0.0001`, `qa:self 1` |
| Accepted update shapes | `profile_scale_calibrated_sequential_profile_stabilization: 1` |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Deterministic verifier | passed with no external model |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Promotion status | rejected for model promotion; safe calibrated movement now spans eight source profiles |

Latest diversity-aware profile-scale floor stabilization screen:

| Signal | Value |
| --- | --- |
| Run | `runs/transformer-answer-v0.95-baseline-floor-diversity-profile-scale-calibrated-sequential-stabilization-configured-step1-dim4-context80/` |
| Mode | `branch-context-profile-baseline-floor-diversity-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood` |
| Added mechanic | diversity-aware profile-scale memory: accept profile-scale updates only when they preserve the baseline floor and do not regress branch-diversity score from the profile's pre-update state |
| Unit coverage | focused transformer tests pass; the mode records diversity activation, attempts, score improvements/ties/regressions, floor rejections, rejection reasons, and accepted profile outcomes |
| Search scales | `1`, `0.25`, `0.05`, `0.01`, `0.0025`, `0.0005`, `0.0001` |
| Outer guard | checked `1/1` step; attempted `1` update; accepted `1`; rejected `0`; outer diversity rejections `0` |
| Profile-scale attempts | `58` attempted; `5` accepted; `53` rejected |
| Diversity outcomes | `5` score improvements; `0` ties; `11` score regressions; `42` floor regressions |
| Accepted profile scales | `bridge:owner 0.0025`, `bridge:place 0.0005`, `fact:learning 0.0005`, `qa:glossary 1`, `qa:learning 0.0025` |
| Accepted update shapes | `profile_scale_diversity_calibrated_sequential_profile_stabilization: 1` |
| Branch-context gate | passed across `219/219` semantic records with no ambiguous, colliding, or skipped records |
| Deterministic verifier | passed with no external model |
| Diversity target | failed, `0/9` multi-target profiles passed |
| Promotion status | rejected for model promotion; accepted movement is now explicitly diversity-score non-regressive |

The transformer is not yet promoted as a reliable responder. It is architecture
evidence: a from-scratch attention model can update weights on the admitted
corpus and leave a checkpoint plus metrics. v0.42 preserves the `37/219`
transformer-only candidate result while improving answer-target NLL versus
v0.41, but raw greedy completion still fails exact answers with the short wrong
completion `" te."`. The latest v0.43 stacked screen proves that two-layer
top-layer-only direct-answer training can complete and write a checkpoint when
the expensive post-direct candidate snapshot is explicitly skipped, but its
repeated `"a"` output is still a failed direct decoder. v0.31's no-candidate
transformer-guided generator remains useful comparison evidence, but it is not
raw transformer decoding. The branch-profile smoke adds a sharper diagnosis:
at the configured branch position, the model is selecting one global token
across prompts instead of separating target-specific answer branches. The
branch-collapse repair uses that diagnosis by penalizing the sampled dominant
branch token, but the evidence shows it only moves the collapse to a new global
token. Branch-batch contrast then trains several distinct target branches in
one update; it lowers loss under sparse dosage, but the branch profile still
collapses globally and even loses the one initially correct QA branch.
`--use-context-mean` then adds a mean-pooled context residual to the final
hidden representation, but the bounded screens still collapse the QA branch to
one wrong global token. The next repair needs a stronger prompt-conditioned
representation signal than simple prompt averaging. `--use-context-projection`
then lets the model learn a zero-initialized projection of that context summary,
and the projection weights do move during training, but the branch profile still
collapses globally. `--use-prompt-attention-summary` makes the summary itself
attention-pooled and trainable, but the bounded screens still collapse globally.
The branch-context coverage diagnostic explains why context-16 branch screens
were partly underdetermined: QA had only four visible branch contexts for eight
records, and those windows mapped to different first target tokens. Context-32
removes literal QA ambiguity but still truncates semantic prompt features.
Context-80 gives every current eval record complete semantic branch-context
coverage with no ambiguity. The next repair needs efficient longer-context
prompt-specific discrimination, not just suppression, batching, or a trainable
summary of a truncated context. The optional branch-context gate now enforces
that distinction for direct-answer screens: unsafe context-16 branch repair can
be skipped and recorded, while complete context-80 branch repair is allowed to
run. The branch-only snapshot mode keeps those longer-context screens practical
by skipping greedy completion evals while still recording the branch diagnostics
and gate evidence needed for the next decision. The first dim8 follow-ups show
that lower branch loss and complete branch context are still not enough: both
repair/contrast and branch-batch contrast collapse QA branch prediction to one
global token. A full greedy-eval promotion snapshot is not warranted until a
screen improves prompt-specific branch diversity. The branch-diversity target
now makes that requirement machine-readable in every direct-answer snapshot.
`branch-diversity-unlikelihood` trains directly against the observed collapse
token and improves the tiny unit case, but the first corpus smoke only moves the
dominant global prediction. Freezing the output bias removes one cheap global
escape hatch, but the corpus smoke still rotates to a single dominant branch
token. Restricted target-set softmax briefly raises QA predicted diversity to
two tokens, then collapses back by the final snapshot. The next repair needs to
make diversity stable across prompts, not just rotate or momentarily crack the
collapsed token. Best-snapshot restoration can preserve a better measured
branch state, but it still ends as a one-token collapse until the underlying
representation separates prompts. Prompt-prefix projection gives the model a
targeted trainable prompt path and the new parameters move, but the evidence
still ends in the same all-`"u"` branch collapse.
Prompt-position projection keeps position-specific prompt access and moves many
more parameters, but the branch profile remains collapsed too.
Branch-target margin adds pairwise target separation on top of that prompt path
and lowers bounded train loss, but the restored branch profile remains the same
one-token collapse.
Branch-representation contrast exposes that the hidden states themselves remain
nearly indistinguishable at the answer branch, so the next repair needs a
stronger prompt-conditioned representation path rather than another output-head
loss alone.
The dim-8 capacity screen increases measured hidden distance, but branch
predictions still collapse globally, so width alone is not the missing repair.
Prompt-position projection scaling shows the prompt residual can be made louder
and the restored hidden-state distance can rise, but the branch prediction
still collapses globally. The pre-layer-norm/final-normalization path is now
implemented and screened; it cracks full collapse in most multi-target profiles
but leaves QA and heldout collapsed. Target-balanced branch batching then
regresses to a baseline-restored global `"n"` collapse, so the next repair
should strengthen prompt-to-answer binding for QA and heldout rather than rely
on sampler balancing or another unrelated loss term. The branch-rank diagnostic
confirms the correct target is usually buried outside the top five predictions,
which points the next repair toward output-head prompt binding instead of a
simple near-miss margin tweak. The first output-binding repair combines that
target-set pressure with representation contrast and improves average target
rank/top-5 evidence, but it still fails target-token coverage and collapses to
wrong branch tokens. The next repair needs to promote the correct target into
the top branch set, not only move it upward while the wrong tokens remain on
top. Hard rank-margin repair is the first screen to make that movement clear:
it lifts correct targets into the top five more often and improves target-token
coverage, but it still leaves a single global wrong prediction. The next repair
needs to convert rank lift into prompt-specific top-1 branch choices. Target-
balanced rank-margin adds some wrong-token diversity and better QA top-3
coverage, but it still does not make correct target tokens win the branch. The
top-one hard-negative screen then regresses rank and top-k coverage, so the
next repair should not simply concentrate more pressure on the current top
wrong token. It needs a prompt-conditioned mechanism that selects among
near-tied branch candidates.

The v0.66 open-source mechanics audit reframes the current blocker as trainer
mechanics rather than another global branch loss. v0.67 implements the first
profile-aware replay-plan surface: branch records carry source/profile keys,
deficits and preservation are computed per profile, and the plan is written as
a run artifact before training.
v0.68 proves that constraint is doing useful work: profile-aware training moved
correct targets upward in the ranked list, but only by collapsing target-token
coverage and branch diversity, so the snapshot gate restored baseline. The next
trainer change needs anti-collapse preservation inside the profile-aware plan.
v0.81 implements that trainer change as a profile target-share objective
mechanic. v0.82 screens it and rejects the trained snapshots because rank lift
still comes from branch collapse. v0.83 adds prompt-specific sibling-target
ownership margins and proves the focused mechanic, but the screen still
restores step `0` because trained snapshots lose target-token coverage. v0.84
anchors replay preservation to baseline predictions and improves trained
coverage relative to v0.83, but still restores step `0` because snapshots miss
the full coverage floor. v0.85 adds a baseline-floor update guard that preserves
the floor by rejecting all attempted unsafe updates. v0.86 retries those updates
at four smaller scales and still rejects every attempt. v0.87 adds one
baseline-covered repair after each failed retry and still rejects every attempt;
v0.88 moves floor anchors into the objective and still rejects every attempt;
v0.89 removes branch pressure and still rejects every floor-stabilization
attempt. v0.90 records the rejected profile floors directly, showing `heldout`
violates every attempt and the worst deficit is `0.25` on `learning`. v0.91
covers the full profile-target floor surface and still rejects every attempt.
v0.92 changes the repair shape to sequential source-profile batches and still
rejects every profile-local attempt. v0.93 calibrates that movement below
`0.01` and accepts one source-profile update at scale `0.0025`. v0.94 adds
profile-scale memory and accepts eight source-profile updates. v0.95 adds
diversity-aware profile-scale acceptance, preserves five score-improving
source-profile updates, and rejects eleven floor-preserving score regressions,
so the next repair should turn non-regressive movement into full branch-diverse
coverage.
