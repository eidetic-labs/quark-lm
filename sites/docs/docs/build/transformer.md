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
on sampler balancing or another unrelated loss term.
